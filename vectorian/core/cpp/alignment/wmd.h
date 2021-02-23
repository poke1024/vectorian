#include "common.h"

struct WMDOptions {
	bool normalize_bow;
	bool symmetric;
	bool one_target;
};

template<typename Index, typename WordId>
class WMD {
public:
	struct RefToken {
		WordId word_id;
		Index i; // index in s or t
		int8_t j; // 0 for s, 1 for t
	};

	struct VocabPair {
		Index vocab;
		Index i;
	};

	struct DistanceRef {
		Index i;
		float d;

		inline bool operator<(const DistanceRef &other) const {
			// inverted, so that heap yields smallest elements.
			return d > other.d;
		}
	};

	struct Document {
		//Eigen::Array<float, Eigen::Dynamic, 1> w1;
		std::vector<float> bow; // (n)bow
		Index w_sum;
		std::vector<VocabPair> vocab;
		// Eigen::Array<Index, 2, Eigen::Dynamic> pos_to_vocab;
		std::vector<Index> pos_to_vocab;
		std::vector<Index> vocab_to_pos; // not correct, since 1:n

		void resize(const size_t p_size) {
			bow.resize(p_size);
			vocab.reserve(p_size);
			pos_to_vocab.resize(p_size);
			vocab_to_pos.resize(p_size);
		}
	};

	size_t m_size;

	std::vector<RefToken> m_tokens;
	Document m_doc[2]; // s, t

	std::vector<float> m_dist;
	std::vector<DistanceRef> m_candidates;
	std::vector<Index> m_result;

	std::vector<Index> m_match;

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		const size_t size = max_len_s + max_len_t;

		m_size = size;
		m_tokens.resize(size);

		for (int i = 0; i < 2; i++) {
			m_doc[i].resize(size);
		}

		m_dist.resize(size * size);
		m_candidates.reserve(size);
		m_result.resize(size);

		m_match.resize(max_len_t);
	}

	inline void reset(const int k) {
		for (int j = 0; j < 2; j++) {
			float *w = m_doc[j].bow.data();
			for (int i = 0; i < k; i++) {
				w[i] = 0.0f;
			}
		}
	}

	template<typename Slice, typename MakeWordId>
	inline int init(
		const Slice &slice,
		const MakeWordId &make_word_id,
		const int len_s, const int len_t,
		const WMDOptions &p_options) {

		int k = 0;
		std::vector<RefToken> &z = m_tokens;

		for (int i = 0; i < len_s; i++) {
			z[k++] = RefToken{
				make_word_id(slice.s(i)), static_cast<Index>(i), 0};
		}
		for (int i = 0; i < len_t; i++) {
			z[k++] = RefToken{
				make_word_id(slice.t(i)), static_cast<Index>(i), 1};
		}

		if (k < 1) {
			return 0;
		}

		std::sort(z.begin(), z.begin() + k, [] (const RefToken &a, const RefToken &b) {
			return a.word_id < b.word_id;
		});

		reset(k);

		for (int i = 0; i < 2; i++) {
			m_doc[i].w_sum = 0;
			m_doc[i].vocab.clear();
		}

		auto cur_word_id = m_tokens[0].word_id;
		int vocab = 0;
		int vocab_mask = 0;

		for (int i = 0; i < k; i++) {
			const auto &token = m_tokens[i];
			const auto new_word_id = token.word_id;

			if (new_word_id != cur_word_id) {
				cur_word_id = new_word_id;
				vocab += 1;
				vocab_mask = 0;
			}

			const int j = token.j;
			auto &doc = m_doc[j];

			doc.bow[vocab] += 1.0f;
			doc.w_sum += 1;
			doc.pos_to_vocab[token.i] = vocab;
			doc.vocab_to_pos[vocab] = token.i;

			if ((vocab_mask & (1 << j)) == 0) {
				doc.vocab.emplace_back(
					VocabPair{
						static_cast<Index>(vocab),
						token.i});
			}

			vocab_mask |= 1 << j;
		}

		if (p_options.normalize_bow) {
			for (int c = 0; c < 2; c++) {
				float *w = m_doc[c].bow.data();
				const float s = m_doc[c].w_sum;
				for (const auto &u : m_doc[c].vocab) {
					w[u.vocab] /= s;
				}
			}
		}

		return vocab + 1;
	}

	template<typename Similarity>
	inline void compute_dist(
		const int len_s, const int len_t,
		const int p_size, const Similarity &sim) {

		Document &doc_s = m_doc[0];
		Document &doc_t = m_doc[1];

		float *dist = m_dist.data();

#if 0
		// since wmd_relaxed will only access dist entries
		// that are sourced from vocab_s and vocab_t, we do
		// not need to initialize the full matrix, which saves
		// us from quadratic time here.

		for (int i = 0; i < p_size; i++) {
			for (int j = 0; j < p_size; j++) {
				dist[i * p_size + j] = 1.0f;
			}
		}
#endif

		for (const auto &u : doc_s.vocab) {
			for (const auto &v : doc_t.vocab) {
				const float d = 1.0f - sim(u.i, v.i);
				dist[u.vocab * p_size + v.vocab] = d;
				dist[v.vocab * p_size + u.vocab] = d;
			}
		}
	}

	struct Weights {
		const float *w;
		float sum;
		const std::vector<VocabPair> *v;
	};

	// inspired by implementation in https://github.com/src-d/wmd-relax
	float _relaxed(
		const int len_s, const int len_t,
		const int size,
		const WMDOptions &p_options) {

		constexpr float max_dist = 1; // assume max dist of 1

		Document &doc_s = m_doc[0];
		Document &doc_t = m_doc[1];
		const Document * const docs[2] = {&doc_t, &doc_s};

		m_match.resize(len_t);

		float cost = 0;
		for (int c = 0; c < 2; c++) {
			const float *w1 = docs[c]->bow.data();
			const float *w2 = docs[1 - c]->bow.data();
			const std::vector<VocabPair> &v1 = docs[c]->vocab;
			const std::vector<VocabPair> &v2 = docs[1 - c]->vocab;

			float acc = 0;
			for (const auto &v1_entry : v1) {
				const int i = v1_entry.vocab;

				if (p_options.one_target) {
					// 1:1 case

					float best_dist = std::numeric_limits<float>::max();
					int best_j = -1;

					// find argmin.
					for (const auto &v2_entry : v2) {
						const int j = v2_entry.vocab;
						const float d = m_dist[i * size + j];
						if (d < best_dist) {
							best_dist = d;
							best_j = j;
						}
					}

					// move w1[i] completely to w2[j].
					const float d = (best_j >= 0 ? best_dist : max_dist);
					acc += w1[i] * d;
					m_result[i] = best_j;

				} else {
					// 1:n case

					float remaining = w1[i];

					auto &candidates = m_candidates;
					candidates.clear();

					for (const auto &v2_entry : v2) {
						const int j = v2_entry.vocab;
						const float d = m_dist[i * size + j];
						candidates.push_back(DistanceRef{
							static_cast<Index>(j),
							d});
					}
					std::make_heap(candidates.begin(), candidates.end());

					while (!candidates.empty()) {
						std::pop_heap(candidates.begin(), candidates.end());
						const auto &r = candidates.back();
						const int w = r.i;

						if (remaining <= w2[w]) {
							acc += remaining * r.d;
							break;
						} else {
							remaining -= w2[w];
							acc += w2[w] * r.d;
						}

						candidates.pop_back();
					}

					if (remaining > 0.0f) {
						acc += remaining * max_dist;
					}
				}
			}

			if (!p_options.normalize_bow) {
				acc /= docs[c]->w_sum;
			}

			if (c == 0) { // w1 is t
				// vocab item i to query pos.
				for (int i = 0; i < len_t; i++) {
					const int j = m_result[doc_t.pos_to_vocab[i]];
					m_match[i] = doc_s.vocab_to_pos[j]; // not ideal
				}
			} else { // w1 is s
				// FIXME
			}

			if (!p_options.symmetric) {
				cost = acc;
				break;
			} else {
				cost = std::max(cost, acc);
			}
		}

		return cost;
	}

	template<typename Slice>
	inline void print_debug(
		const QueryRef &p_query, const Slice &slice,
		const int len_s, const int len_t, const int vocabulary_size,
		std::ostream &os) {

		const QueryVocabularyRef vocab = p_query->vocabulary();

		os << "s: ";
		for (int i = 0; i < len_s; i++) {
			os << vocab->id_to_token(slice.s(i).id) << " [" << slice.s(i).id << "]" <<  " ";
		}
		os << "\n";

		os << "t: ";
		for (int i = 0; i < len_t; i++) {
			os << vocab->id_to_token(slice.t(i).id) << " [" << slice.t(i).id << "]" <<  " ";
		}
		os << "\n";

		os << "vocab s: ";
		for (const auto &u : m_doc[0].vocab) {
			os << vocab->id_to_token(slice.s(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "vocab t: ";
		for (const auto &u : m_doc[1].vocab) {
			os << vocab->id_to_token(slice.t(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "w1: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_doc[0].bow[i] << " ";
		}
		os << "\n";

		os << "w2: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_doc[1].bow[i] << " ";
		}
		os << "\n";

		os << "dist: \n";
		for (int i = 0; i < vocabulary_size; i++) {
			for (int j = 0; j < vocabulary_size; j++) {
				os << m_dist[i * vocabulary_size + j] << " ";
			}
			os << "\n";
		}
	}

	template<typename Slice, typename MakeWordId>
	float relaxed(
		const Slice &slice,
		const size_t len_s,
		const size_t len_t,
		const MakeWordId &make_word_id,
		const WMDOptions &p_options,
		const float max_weighted_score) {

		if (p_options.symmetric && !p_options.normalize_bow) {
			throw std::runtime_error(
				"cannot run symmetric mode WMD with bow (needs nbow)");
		}

		const float max_score = p_options.normalize_bow ?
			1.0f : max_weighted_score;

		const int vocabulary_size = init(
			slice, make_word_id, len_s, len_t, p_options);

		if (vocabulary_size == 0) {
			return 0.0f;
		}

		compute_dist(
			len_s, len_t,
			vocabulary_size,
			[&slice] (int i, int j) -> float {
				return slice.similarity(i, j);
			});

		/*std::ofstream outfile;
		outfile.open("/Users/arbeit/Desktop/debug_wmd.txt", std::ios_base::app);
		print_debug(p_query, slice, len_s, len_t, vocabulary_size, outfile);*/

		//p_query->t_tokens_pos_weights();

		return max_score - _relaxed(
			len_s, len_t,
			vocabulary_size,
			p_options);
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &match() {
		return m_match;
	}
};