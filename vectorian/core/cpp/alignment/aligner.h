#ifndef __VECTORIAN_ALIGNER__
#define __VECTORIAN_ALIGNER__

// Author: Bernhard Liebl, 2020
// Released under a MIT license.

// Aligner always computes one best alignment, but there
// might be multiple such alignments.

#include <xtensor/xsort.hpp>

template<
	typename Index=int16_t,
	typename SimilarityScore=float>

class Aligner {
private:
	class Fold {
	private:
		SimilarityScore m_score;
		std::pair<Index, Index> m_traceback;

	public:
		inline Fold(const SimilarityScore zero_score) :
			m_score(zero_score),
			m_traceback(std::make_pair(-1, -1)) {
		}

		inline Fold(
			const SimilarityScore score,
			const std::pair<Index, Index> &traceback) :

			m_score(score),
			m_traceback(traceback) {

		}

		inline void update(
			const SimilarityScore score,
			const std::pair<Index, Index> &traceback) {

			if (score > m_score) {
				m_score = score;
				m_traceback = traceback;
			}
		}

		inline SimilarityScore score() const {
			return m_score;
		}

		inline const std::pair<Index, Index> &traceback() const {
			return m_traceback;
		}
	};

	const size_t m_max_len_s;
	const size_t m_max_len_t;

	struct TracebackMatrix {
		xt::xtensor<Index, 3> matrix;

		struct ConstElement {
			const xt::xtensor<Index, 3> &matrix;
			const Index i;
			const Index j;

			inline std::pair<Index, Index> to_pair() const {
				return std::make_pair(matrix(i, j, 0), matrix(i, j, 1));
			}

			inline operator std::pair<Index, Index>() const {
				return to_pair();
			}
		};

		struct Element {
			xt::xtensor<Index, 3> &matrix;
			const Index i;
			const Index j;

			inline std::pair<Index, Index> to_pair() const {
				return std::make_pair(matrix(i, j, 0), matrix(i, j, 1));
			}

			inline operator std::pair<Index, Index>() const {
				return to_pair();
			}

			inline Element &operator=(const std::pair<Index, Index> &value) {
				matrix(i, j, 0) = value.first;
				matrix(i, j, 1) = value.second;
				return *this;
			}
		};

		inline ConstElement operator()(const Index i, const Index j) const {
			return ConstElement{matrix, i, j};
		}

		inline Element operator()(const Index i, const Index j) {
			return Element{matrix, i, j};
		}
	};

	xt::xtensor<SimilarityScore, 2> m_values;
	TracebackMatrix m_traceback;
	xt::xtensor<Index, 1> m_best_column;

	SimilarityScore m_best_score;
	std::vector<Index> m_best_match;

	template<typename Flow>
	inline bool reconstruct_local_alignment(
		Flow &flow,
		const Index len_t,
		const Index len_s,
		const SimilarityScore zero_similarity) {

		const auto &values = m_values;
		const auto &traceback = m_traceback;

		xt::view(m_best_column, xt::range(0, len_s)) = xt::argmax(
			xt::view(values, xt::range(0, len_s), xt::range(0, len_t)), 1);

		SimilarityScore score = 0.0f;
		Index best_u = 0, best_v = 0;

		for (Index u = 0; u < len_s; u++) {
			const Index v = m_best_column[u];
			const SimilarityScore s = values(u, v);
			if (s > score) {
				score = s;
				best_u = u;
				best_v = v;
			}
		}

		if (score <= zero_similarity) {
			m_best_score = 0.0f;
			return false;
		}

		m_best_score = score;

		flow.initialize(len_t);
		//_best_match.resize(len_t);
		//std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = best_u;
		Index v = best_v;
		while (u >= 0 && v >= 0 && values(u, v) > zero_similarity) {
			flow.set(v, u);
			//_best_match[v] = u;
			std::tie(u, v) = traceback(u, v).to_pair();
		}

		return true;
	}

	template<typename Flow>
	inline void reconstruct_global_alignment(
		Flow &flow,
		const Index len_t,
		const Index len_s) {

		flow.initialize(len_t);
		//_best_match.resize(len_t);
		//std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = len_s - 1;
		Index v = len_t - 1;
		m_best_score = m_values(u, v);

		while (u >= 0 && v >= 0) {
			flow.set(v, u);
			//_best_match[v] = u;
			std::tie(u, v) = m_traceback(u, v).to_pair();
		}
	}

public:
	Aligner(Index max_len_s, Index max_len_t) :
		m_max_len_s(max_len_s),
		m_max_len_t(max_len_t) {

		m_values.resize({static_cast<size_t>(max_len_s), static_cast<size_t>(max_len_t)});
		m_traceback.matrix.resize({static_cast<size_t>(max_len_s), static_cast<size_t>(max_len_t), 2});
		m_best_column.resize({static_cast<size_t>(max_len_s)});
	}

	auto value_matrix(Index len_s, Index len_t) {
		return xt::view(m_values, xt::range(0, len_s), xt::range(0, len_t));
	}

	auto traceback_matrix(Index len_s, Index len_t) {
		return xt::view(m_traceback.matrix, xt::range(0, len_s), xt::range(0, len_t));
	}

	inline SimilarityScore score() const {
		return m_best_score;
	}

#if 0 && !defined(ALIGNER_SLIM)
	std::string pretty_printed(
		const std::string &s,
		const std::string &t) {

		std::ostringstream out[3];

		int i = 0;
		for (int j = 0; j < t.length(); j++) {
			auto m = _best_match[j];
			if (m < 0) {
				out[0] << "-";
				out[1] << " ";
				out[2] << t[j];
			} else {
				while (i < m) {
					out[0] << s[i];
					out[1] << " ";
					out[2] << "-";
					i += 1;
				}

				out[0] << s[m];
				out[1] << "|";
				out[2] << t[j];
				i = m + 1;
			}
		}

		while (i < s.length()) {
			out[0] << s[i];
			out[1] << " ";
			out[2] << "-";
			i += 1;
		}

		std::ostringstream r;
		r << out[0].str() << "\n" << out[1].str() << "\n" << out[2].str();
		return r.str();
	}
#endif

	template<typename Flow, typename Similarity>
	void needleman_wunsch(
		Flow &flow,
		const Similarity &similarity,
		const SimilarityScore gap_cost, // linear
		const Index len_s,
		const Index len_t) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > m_max_len_t || size_t(len_s) > m_max_len_s) {
			throw std::invalid_argument("len larger than max");
		}

		auto &values = m_values;
		auto &traceback = m_traceback;

		const auto nwvalues = [&values, &gap_cost] (Index u, Index v) {
			if (u >= 0 && v >= 0) {
				return values(u, v);
			} else if (u < 0) {
				return -gap_cost * (v + 1);
			} else {
				return -gap_cost * (u + 1);
			}
		};

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				const SimilarityScore s0 =
					nwvalues(u - 1, v - 1);
				const SimilarityScore s1 =
					similarity(u, v);
				Fold best(
					s0 + s1,
					std::make_pair(u - 1, v - 1));

				best.update(
					nwvalues(u - 1, v) - gap_cost,
					std::make_pair(u - 1, v));

				best.update(
					nwvalues(u, v - 1) - gap_cost,
					std::make_pair(u, v - 1));

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_global_alignment(flow, len_t, len_s);
	}

	template<typename Flow, typename Similarity>
	void smith_waterman(
		Flow &flow,
		const Similarity &similarity,
		const SimilarityScore gap_cost, // linear
		const Index len_s,
		const Index len_t,
		const SimilarityScore zero_similarity = 0) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > m_max_len_t || size_t(len_s) > m_max_len_s) {
			throw std::invalid_argument("len larger than max");
		}

		auto &values = m_values;
		auto &traceback = m_traceback;

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				Fold best(zero_similarity);

				{
					const SimilarityScore s0 =
						(v > 0 && u > 0) ? values(u - 1, v - 1) : 0;
					const SimilarityScore s1 =
						similarity(u, v);
					best.update(
						s0 + s1,
						std::make_pair(u - 1, v - 1));
				}

				if (u > 0) {
					best.update(
						values(u - 1, v) - gap_cost,
						std::make_pair(u - 1, v));
				}

				if (v > 0) {
					best.update(
						values(u, v - 1) - gap_cost,
						std::make_pair(u, v - 1));
				}

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_local_alignment(flow, len_t, len_s, zero_similarity);
	}

	template<typename Flow, typename Similarity, typename Gap>
	void waterman_smith_beyer(
		Flow &flow,
		const Similarity &similarity,
		const Gap &gap_cost,
		const Index len_s,
		const Index len_t,
		const SimilarityScore zero_similarity = 0) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > m_max_len_t || size_t(len_s) > m_max_len_s) {
			throw std::invalid_argument("len larger than max");
		}

		auto &values = m_values;
		auto &traceback = m_traceback;

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				Fold best(zero_similarity);

				{
					const SimilarityScore s0 =
						(v > 0 && u > 0) ? values(u - 1, v - 1) : 0;
					const SimilarityScore s1 =
						similarity(u, v);
					best.update(
						s0 + s1,
						std::make_pair(u - 1, v - 1));
				}

				for (Index k = 0; k < u; k++) {
					best.update(
						values(k, v) - gap_cost(u - k),
						std::make_pair(k, v));
				}

				for (Index k = 0; k < v; k++) {
					best.update(
						values(u, k) - gap_cost(v - k),
						std::make_pair(u, k));
				}

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_local_alignment(flow, len_t, len_s, zero_similarity);
	}
};

#endif // __VECTORIAN_ALIGNER__
