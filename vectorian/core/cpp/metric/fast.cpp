#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "match/matcher_impl.h"

#include <fstream>

template<typename Index>
class WatermanSmithBeyer {
	std::shared_ptr<Aligner<Index, float>> m_aligner;
	const std::vector<float> m_gap_cost;
	const float m_smith_waterman_zero;

public:
	WatermanSmithBeyer(
		const std::vector<float> &p_gap_cost,
		float p_zero=0.5) :

		m_gap_cost(p_gap_cost),
		m_smith_waterman_zero(p_zero) {

		PPK_ASSERT(m_gap_cost.size() >= 1);
	}

	void init(Index max_len_s, Index max_len_t) {
		m_aligner = std::make_shared<Aligner<Index, float>>(
			max_len_s, max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return m_gap_cost[
			std::min(len, m_gap_cost.size() - 1)];
	}

	template<typename Slice>
	inline void operator()(
		const QueryRef &, const Slice &slice, const int len_s, const int len_t) const {

		m_aligner->waterman_smith_beyer(
			[&slice] (int i, int j) -> float {
				return slice.score(i, j);
			},
			[this] (size_t len) -> float {
				return this->gap_cost(len);
			},
			len_s,
			len_t,
			m_smith_waterman_zero);
	}

	inline float score() const {
		return m_aligner->score();
	}

	inline const std::vector<Index> &match() const {
		return m_aligner->match();
	}

	inline std::vector<Index> &mutable_match() {
		return m_aligner->mutable_match();
	}
};

template<typename Index>
class RelaxedWordMoversDistance {
	const bool m_normalize_bow;
	const bool m_symmetric;
	const bool m_one_target;

	float m_score;
	std::vector<Index> m_match;

	struct RefToken {
		wvec_t word_id;
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

	struct Scratch {
		size_t size;

		std::vector<RefToken> tokens;
		//Eigen::Array<float, Eigen::Dynamic, 1> w1;
		//Eigen::Array<float, Eigen::Dynamic, 1> w2;
		std::vector<float> w1; // (n)bow for s
		std::vector<float> w2; // (n)bow for t
		std::vector<VocabPair> vocab_s;
		std::vector<VocabPair> vocab_t;

		// Eigen::Array<Index, 2, Eigen::Dynamic> pos_to_vocab;
		std::vector<Index> pos_to_vocab_s;
		std::vector<Index> pos_to_vocab_t;

		std::vector<Index> vocab_to_pos_s; // not correct, since 1:n
		std::vector<Index> vocab_to_pos_t; // not correct, since 1:n

		std::vector<float> dist;
		std::vector<DistanceRef> candidates;
		std::vector<Index> result;

		void resize(const size_t p_size) {
			size = p_size;
			tokens.resize(p_size);
			w1.resize(p_size);
			w2.resize(p_size);
			vocab_s.reserve(p_size);
			vocab_t.reserve(p_size);

			pos_to_vocab_s.resize(p_size);
			pos_to_vocab_t.resize(p_size);
			vocab_to_pos_s.resize(p_size);
			vocab_to_pos_t.resize(p_size);

			dist.resize(p_size * p_size);
			candidates.reserve(p_size);
			result.resize(p_size);
		}

		inline void reset(const int k) {
			for (int i = 0; i < k; i++) {
				w1[i] = 0.0f;
				w2[i] = 0.0f;
			}
		}

		inline int init_for_k(const int k, const bool normalize_bow) {
			reset(k);

			float * const ws[2] = {
				w1.data(),
				w2.data()
			};

			Index w_sum[2] = {
				0, 0
			};

			vocab_s.clear();
			vocab_t.clear();

			std::vector<VocabPair> * const vocab_x[2] = {
				&vocab_s,
				&vocab_t
			};

			std::vector<Index> * const pos_to_vocab_x[2] = {
				&pos_to_vocab_s,
				&pos_to_vocab_t
			};
			std::vector<Index> * const vocab_to_pos_x[2] = {
				&vocab_to_pos_s,
				&vocab_to_pos_t
			};

			int cur_word_id = tokens[0].word_id;
			int vocab = 0;
			int vocab_mask = 0;

			for (int i = 0; i < k; i++) {
				const auto &token = tokens[i];
				const int new_word_id = token.word_id;

				if (new_word_id != cur_word_id) {
					cur_word_id = new_word_id;
					vocab += 1;
					vocab_mask = 0;
				}

				const int j = token.j;
				ws[j][vocab] += 1.0f;
				w_sum[j] += 1;

				if ((vocab_mask & (1 << j)) == 0) {
					vocab_x[j]->emplace_back(
						VocabPair{
							static_cast<Index>(vocab),
							token.i});
				}

				(*pos_to_vocab_x[j])[token.i] = vocab;
				(*vocab_to_pos_x[j])[vocab] = token.i;

				vocab_mask |= 1 << j;
			}

			if (normalize_bow) {
				for (int c = 0; c < 2; c++) {
					float *w = ws[c];
					const float s = w_sum[c];
					for (const auto &u : *vocab_x[c]) {
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

			for (const auto &u : vocab_s) {
				for (const auto &v : vocab_t) {
					const float d = 1.0f - sim(u.i, v.i);
					dist[u.vocab * p_size + v.vocab] = d;
					dist[v.vocab * p_size + u.vocab] = d;
				}
			}
		}
	};

	Scratch m_scratch;

	// inspired by implementation in https://github.com/src-d/wmd-relax
	template <typename T>
	T wmd_relaxed(
		const T *w1, const T *w2,
		const std::vector<VocabPair> *v1,
		const std::vector<VocabPair> *v2,
		const T *dist, const int size, const int len_t, const int len_s) {

		constexpr float max_dist = 1; // assume max dist of 1

		T cost = 0;
		for (int c = 0; c < 2; c++) {
			T acc = 0;
			for (const auto &v1_entry : *v1) {
				const int i = v1_entry.vocab;

				if (m_one_target) {
					// 1:1 case

					float best_dist = std::numeric_limits<float>::max();
					int best_j = -1;

					// find argmin.
					for (const auto &v2_entry : *v2) {
						const int j = v2_entry.vocab;
						const float d = dist[i * size + j];
						if (d < best_dist) {
							best_dist = d;
							best_j = j;
						}
					}

					// move w1[i] completely to w2[j].
					if (best_j >= 0) {
						acc += w1[i] * best_dist;
					} else {
						acc += w1[i] * max_dist;
					}
					m_scratch.result[i] = best_j;

				} else {
					// 1:n case

					T remaining = w1[i];

					auto &candidates = m_scratch.candidates;
					candidates.clear();

					for (const auto &v2_entry : *v2) {
						const int j = v2_entry.vocab;
						const float d = dist[i * size + j];
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

			if (c == 0) { // w1 is t
				// vocab item i to query pos.
				for (int i = 0; i < len_t; i++) {
					const int j = m_scratch.result[m_scratch.pos_to_vocab_t[i]];
					m_match[i] = m_scratch.vocab_to_pos_s[j]; // not ideal
				}
			} else { // w1 is s
				// FIXME
			}

			if (!m_symmetric) {
				cost = acc;
				break;
			} else {
				cost = std::max(cost, acc);
				std::swap(w1, w2);
				std::swap(v1, v2);
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
		for (const auto &u : m_scratch.vocab_s) {
			os << vocab->id_to_token(slice.s(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "vocab t: ";
		for (const auto &u : m_scratch.vocab_t) {
			os << vocab->id_to_token(slice.t(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "w1: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_scratch.w1[i] << " ";
		}
		os << "\n";

		os << "w2: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_scratch.w2[i] << " ";
		}
		os << "\n";

		os << "dist: \n";
		for (int i = 0; i < vocabulary_size; i++) {
			for (int j = 0; j < vocabulary_size; j++) {
				os << m_scratch.dist[i * vocabulary_size + j] << " ";
			}
			os << "\n";
		}

		os << "\n";
	}

public:
	RelaxedWordMoversDistance(
		const bool p_normalize_bow,
		const bool p_symmetric,
		const bool p_one_target) :

		m_normalize_bow(p_normalize_bow),
		m_symmetric(p_symmetric),
		m_one_target(p_one_target) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_scratch.resize(max_len_s + max_len_t);
		m_match.reserve(max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline void operator()(
		const QueryRef &p_query, const Slice &slice, const int len_s, const int len_t) {

		int k = 0;
		std::vector<RefToken> &z = m_scratch.tokens;
		const auto &enc = slice.encoder();

		for (int i = 0; i < len_s; i++) {
			z[k++] = RefToken{
				enc.to_embedding(slice.s(i)), static_cast<Index>(i), 0};
		}
		for (int i = 0; i < len_t; i++) {
			z[k++] = RefToken{
				enc.to_embedding(slice.t(i)), static_cast<Index>(i), 1};
		}

		if (k < 1) {
			m_score = 0;
			return;
		}

		std::sort(z.begin(), z.begin() + k, [] (const RefToken &a, const RefToken &b) {
			return a.word_id < b.word_id;
		});

		const int vocabulary_size = m_scratch.init_for_k(k, m_normalize_bow);

		m_scratch.compute_dist(len_s, len_t, vocabulary_size, [&slice] (int i, int j) -> float {
			return slice.similarity(i, j);
		});

		std::ofstream outfile;
		outfile.open("/Users/arbeit/Desktop/debug_wmd.txt", std::ios_base::app);
		print_debug(p_query, slice, len_s, len_t, vocabulary_size, outfile);

		m_match.resize(len_t);

		m_score = 1.0f - wmd_relaxed<float>(
			m_scratch.w2.data(), // t
			m_scratch.w1.data(), // s
			&m_scratch.vocab_t,
			&m_scratch.vocab_s,
			m_scratch.dist.data(),
			vocabulary_size,
			len_t,
			len_s);

		outfile << "score is " << m_score;
	}

	inline float score() const {
		return m_score;
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &mutable_match() {
		return m_match;
	}
};

template<typename Scores, typename Aligner>
MatcherRef make_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores,
	const Aligner &p_aligner) {

	if (p_query->bidirectional()) {
		return std::make_shared<MatcherImpl<Scores, Aligner, TokenIdEncoder, true>>(
			p_query, p_document, p_metric, p_aligner, scores);
	} else {
		return std::make_shared<MatcherImpl<Scores, Aligner, TokenIdEncoder, false>>(
			p_query, p_document, p_metric, p_aligner, scores);
	}
}

template<typename Scores>
MatcherRef create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores) {

	// FIXME support different alignment algorithms here.

	py::gil_scoped_acquire acquire;

	const py::dict args = p_query->alignment_algorithm();

	std::string algorithm;
	if (args.contains("algorithm")) {
		algorithm = args["algorithm"].cast<py::str>();
	} else {
		algorithm = "wsb"; // default
	}

	if (algorithm == "wsb") {
		float zero = 0.5;
		if (args.contains("zero")) {
			zero = args["zero"].cast<float>();
		}

		std::vector<float> gap_cost;
		if (args.contains("gap")) {
			auto cost = args["gap"].cast<py::array_t<float>>();
			auto r = cost.unchecked<1>();
			const ssize_t n = r.shape(0);
			gap_cost.resize(n);
			for (ssize_t i = 0; i < n; i++) {
				gap_cost[i] = r(i);
			}
		}
		if (gap_cost.empty()) {
			gap_cost.push_back(std::numeric_limits<float>::infinity());
		}

		return make_matcher(
			p_query, p_document, p_metric, scores,
			WatermanSmithBeyer<int16_t>(gap_cost, zero));

	} else if (algorithm == "rwmd") {

		bool normalize_bow = true;
		bool symmetric = true;
		bool one_target = true;

		if (args.contains("normalize_bow")) {
			normalize_bow = args["normalize_bow"].cast<bool>();
		}
		if (args.contains("symmetric")) {
			symmetric = args["symmetric"].cast<bool>();
		}
		if (args.contains("one_target")) {
			one_target = args["one_target"].cast<bool>();
		}

		return make_matcher(
			p_query, p_document, p_metric, scores,
			RelaxedWordMoversDistance<int16_t>(
				normalize_bow, symmetric, one_target));

	} else {

		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
}


MatcherRef FastMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	auto self = std::dynamic_pointer_cast<FastMetric>(shared_from_this());

	std::vector<FastScores> scores;
	scores.emplace_back(FastScores(p_query, p_document, self));

	return ::create_matcher(p_query, p_document, self, scores);
}

const std::string &FastMetric::name() const {
	return m_embedding->name();
}
