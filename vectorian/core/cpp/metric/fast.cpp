#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "match/matcher_impl.h"

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
		const Slice &slice, int len_s, int len_t) const {

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

// from: https://github.com/src-d/wmd-relax/blob/master/emd_relaxed.h
template <typename T>
T emd_relaxed(const T *__restrict__ w1, const T *__restrict__ w2,
              const T *__restrict__ dist, uint32_t size,
              int32_t *boilerplate) {

  for (size_t i = 0; i < size; i++) {
    boilerplate[i] = i;
  }

  T cost = std::numeric_limit<T>::max();
  for (size_t c = 0; c < 1; c++) { // do not flip problem.
    T acc = 0;
    for (size_t i = 0; i < size; i++) {
      if (w1[i] != 0) {
        // FIXME use a heap.

        std::sort(
          boilerplate,
          boilerplate + size,
          [&](const int a, const int b) {
            return dist[i * size + a] < dist[i * size + b];
          });

        T remaining = w1[i];
        for (size_t j = 0; j < size; j++) {
          int w = boilerplate[j];
          if (remaining < w2[w]) {
            acc += remaining * dist[i * size + w];
            break;
          } else {
            remaining -= w2[w];
            acc += w2[w] * dist[i * size + w];
          }
        }
      }
    }
    cost = std::min(cost, acc);
    std::swap(w1, w2);
  }
  return cost;
}

template<typename Index>
class RelaxedWordMoversDistance {
	float m_score;
	std::vector<Index> m_match;

	struct RefToken {
		wvec_t word_id;
		Index i; // index in s or t
		int8_t j; // 0 for s, 1 for t
	};

	struct Scratch {
		size_t size;

		std::vector<RefToken> tokens;
		//Eigen::Array<float, Eigen::Dynamic, 1> w1;
		//Eigen::Array<float, Eigen::Dynamic, 1> w2;
		std::vector<float> w1;
		std::vector<float> w2;
		std::vector<token_t> back_s;
		std::vector<token_t> back_t;
		std::vector<float> dist;
		std::vector<int32_t> match;

		void resize(const size_t p_size) {
			size = p_size;
			tokens.resize(p_size);
			w1.resize(p_size);
			w2.resize(p_size);
			back_s.resize(p_size);
			back_t.resize(p_size);
			dist.resize(p_size * p_size);
			match.resize(p_size);
		}

		inline void reset(const int k) {
			for (int i = 0; i < k; i++) {
				w1[i] = 0.0f;
				w2[i] = 0.0f;
				back_s[i] = -1;
				back_t[i] = -1;
			}
		}

		inline int init_for_k(const int k) {
			reset(k);

			float * const ws[2] = {
				w1.data(),
				w2.data()
			};

			Index wsum[2] = {
				0, 0
			};

			token_t * const back[2] = {
				back_s.data(),
				back_t.data()
			};

			int cur_word_id = tokens[0].word_id;
			int vocab = 0;

			for (int i = 0; i < k; i++) {
				const auto &token = tokens[i];
				const int new_word_id = token.word_id;

				if (new_word_id != cur_word_id) {
					cur_word_id = new_word_id;
					vocab += 1;
				}

				const int j = token.j;
				ws[j][vocab] += 1.0f;
				wsum[j] += 1;
				back[j][vocab] = token.i;
			}

			const float n1 = wsum[0];
			const float n2 = wsum[1];
			for (int i = 0; i < k; i++) {
				w1[i] /= n1;
				w2[i] /= n2;
			}

			return vocab + 1;
		}

		template<typename Similarity>
		inline void compute_dist(const int p_size, const Similarity &sim) {
			for (int i = 0; i < p_size; i++) {
				for (int j = 0; j < p_size; j++) {
					if (i == j) {
						dist[i * p_size + j] = 0.0f;
						continue;
					}

					{
						const int u = back_s[i];
						if (u >= 0) {
							const int v = back_t[j];
							if (v >= 0) {
								dist[i * p_size + j] = 1.0f - std::max(0.0f, sim(u, v));
								continue;
							}
						}
					}

					{
						const int u = back_s[j];
						if (u >= 0) {
							const int v = back_t[i];
							if (v >= 0) {
								dist[i * p_size + j] = 1.0f - std::max(0.0f, sim(u, v));
								continue;
							}
						}
					}

					dist[i * p_size + j] = 1.0f;
				}
			}
		}
	};

	Scratch m_scratch;

public:
	RelaxedWordMoversDistance() {
	}

	void init(Index max_len_s, Index max_len_t) {
		//m_cache.allocate(std::max(max_len_s, max_len_t));
		m_scratch.resize(max_len_s + max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline void operator()(
		const Slice &slice, int len_s, int len_t) {

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

		const int problem_size = m_scratch.init_for_k(k);

		m_scratch.compute_dist(problem_size, [&slice] (int i, int j) -> float {
			return slice.similarity(i, j);
		});

		m_score = 1.0f - emd_relaxed<float>(
			m_scratch.w2.data(), // t
			m_scratch.w1.data(), // s
			m_scratch.dist.data(),
			problem_size,
			m_scratch.match.data());
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

	} else if (algorithm == "relaxed-wmd") {

		return make_matcher(
			p_query, p_document, p_metric, scores,
			RelaxedWordMoversDistance<int16_t>());

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
