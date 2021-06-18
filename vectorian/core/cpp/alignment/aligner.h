#ifndef __VECTORIAN_ALIGNER__
#define __VECTORIAN_ALIGNER__

#include "common.h"

// Aligner always computes one best alignment, but there
// might be multiple such alignments.

#include <xtensor/xsort.hpp>

namespace alignments {

typedef std::function<xt::pyarray<float>(size_t)> GapTensorFactory;

template<typename Index, typename Value>
class Matrix;

template<typename Index, typename Value>
class MatrixFactory {
protected:
	friend class Matrix<Index, Value>;

	struct Data {
		xt::xtensor<Value, 2> values;
		xt::xtensor<Index, 3> traceback;
		xt::xtensor<Index, 1> best_column;
	};

	const std::unique_ptr<Data> m_data;
	const Index m_max_len_s;
	const Index m_max_len_t;

public:
	inline MatrixFactory(
		const Index p_max_len_s,
		const Index p_max_len_t) :

		m_data(std::make_unique<Data>()),
		m_max_len_s(p_max_len_s),
		m_max_len_t(p_max_len_t) {

		m_data->values.resize({
			static_cast<size_t>(m_max_len_s) + 1,
			static_cast<size_t>(m_max_len_t) + 1
		});
		m_data->traceback.resize({
			static_cast<size_t>(m_max_len_s),
			static_cast<size_t>(m_max_len_t),
			2
		});
		m_data->best_column.resize({
			static_cast<size_t>(m_max_len_s)
		});
	}

	inline Matrix<Index, Value> make(
		const Index len_s, const Index len_t) const;

	inline Index max_len() const {
		return std::max(m_max_len_s, m_max_len_t);
	}

	inline auto &values() const {
		return m_data->values;
	}
};

template<typename Index, typename Value>
class Matrix {
	const MatrixFactory<Index, Value> &m_factory;
	const Index m_len_s;
	const Index m_len_t;

public:
	inline Matrix(
		const MatrixFactory<Index, Value> &factory,
		const Index len_s,
		const Index len_t) :

	    m_factory(factory),
	    m_len_s(len_s),
	    m_len_t(len_t) {
	}

	inline Index len_s() const {
		return m_len_s;
	}

	inline Index len_t() const {
		return m_len_t;
	}

	inline auto values() const {
		return xt::view(
			m_factory.m_data->values,
			xt::range(1, m_len_s + 1),
			xt::range(1, m_len_t + 1));
	}

	inline auto traceback() const {
		return xt::view(
			m_factory.m_data->traceback,
			xt::range(0, m_len_s),
			xt::range(0, m_len_t));
	}

	inline auto best_column() const {
		return xt::view(
			m_factory.m_data->best_column,
			xt::range(0, m_len_s));
	}
};

template<typename Index, typename Value>
inline Matrix<Index, Value> MatrixFactory<Index, Value>::make(
	const Index len_s, const Index len_t) const {

	if (len_s > m_max_len_s) {
		throw std::invalid_argument("len of s larger than max");
	}
	if (len_t > m_max_len_t) {
		throw std::invalid_argument("len of t larger than max");
	}

	return Matrix(*this, len_s, len_t);
}


template<typename Value>
class Local {
private:
	const Value m_zero;

public:
	inline Local(const Value p_zero) : m_zero(p_zero) {
	}

	void init_border(
		xt::xtensor<Value, 2> &p_values,
		const xt::xtensor<Value, 1> &p_gap_cost_s,
		const xt::xtensor<Value, 1> &p_gap_cost_t) const {

		xt::view(p_values, xt::all(), 0).fill(0);
		xt::view(p_values, 0, xt::all()).fill(0);
	}

	template<typename Fold>
	inline void update_best(Fold &fold) const {
		fold.update(m_zero, -1, -1);
	}

	template<typename Flow, typename Index>
	inline Value traceback(
		Flow &flow,
		Matrix<Index, Value> &matrix) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.values();
		const auto traceback = matrix.traceback();
		auto best_column = matrix.best_column();

		const auto zero_similarity = m_zero;

		best_column = xt::argmax(values, 1);

		Value score = zero_similarity;
		Index best_u = 0, best_v = 0;

		for (Index u = 0; u < len_s; u++) {
			const Index v = best_column(u);
			const Value s = values(u, v);
			if (s > score) {
				score = s;
				best_u = u;
				best_v = v;
			}
		}

		if (score <= zero_similarity) {
			return 0;
		}

		flow.initialize(len_t);
		//_best_match.resize(len_t);
		//std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = best_u;
		Index v = best_v;

		Index last_u = -1;
		Index last_v = -1;

		while (u >= 0 && v >= 0 && values(u, v) > zero_similarity) {
			if (u == last_u) {
				flow.reset(last_v);
			}

			flow.set(v, u);
			//_best_match[v] = u;
			last_u = u;
			last_v = v;

			const auto t = xt::view(traceback, u, v, xt::all());
			u = t(0);
			v = t(1);
		}

		if (u >= 0 && v >= 0) {
			return score - values(u, v);
		} else {
			return score;
		}
	}
};


template<typename Value>
class Global {
public:
	void init_border(
		xt::xtensor<Value, 2> &p_values,
		const xt::xtensor<Value, 1> &p_gap_cost_s,
		const xt::xtensor<Value, 1> &p_gap_cost_t) const {

		xt::view(p_values, xt::all(), 0) = -1 * xt::view(
			p_gap_cost_s, xt::range(0, p_values.shape(0)));
		xt::view(p_values, 0, xt::all()) = -1 * xt::view(
			p_gap_cost_t, xt::range(0, p_values.shape(1)));
	}

	template<typename Fold>
	inline void update_best(Fold &fold) const {
	}

	template<typename Flow, typename Index>
	inline float traceback(
		Flow &flow,
		Matrix<Index, Value> &matrix) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.values();
		const auto traceback = matrix.traceback();

		flow.initialize(len_t);
		//_best_match.resize(len_t);
		//std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = len_s - 1;
		Index v = len_t - 1;
		const Value best_score = values(u, v);

		Index last_u = -1;
		Index last_v = -1;

		while (u >= 0 && v >= 0) {
			if (u == last_u) {
				flow.reset(last_v);
			}

			flow.set(v, u);
			//_best_match[v] = u;
			last_u = u;
			last_v = v;

			const auto t = xt::view(traceback, u, v, xt::all());
			u = t(0);
			v = t(1);
		}

		return best_score;
	}
};


template<typename Index, typename Value>
class MaxFold {
public:
	typedef xt::xtensor_fixed<Index, xt::xshape<2>> Coord;

private:
	Value m_score;
	Coord m_traceback;

public:
	inline MaxFold() {
	}

	inline MaxFold(
		const Value score,
		const Coord &traceback) :

		m_score(score),
		m_traceback(traceback) {

	}

	inline void set(
		const Value score,
		const Index u,
		const Index v) {

		m_score = score;
		m_traceback[0] = u;
		m_traceback[1] = v;
	}

	inline void update(
		const Value score,
		const Index u,
		const Index v) {

		if (score > m_score) {
			m_score = score;
			m_traceback[0] = u;
			m_traceback[1] = v;
		}
	}

	inline Value score() const {
		return m_score;
	}

	inline const Coord &traceback() const {
		return m_traceback;
	}
};



template<typename Locality, typename Index=int16_t, typename Value=float>
class Aligner {
protected:
	const Locality m_locality;
	MatrixFactory<Index, Value> m_factory;

public:
	inline Aligner(
		const Locality &p_locality,
		const Index p_max_len_s,
		const Index p_max_len_t) :

		m_locality(p_locality),
		m_factory(p_max_len_s, p_max_len_t) {
	}

	inline Index max_len() const {
		return m_factory.max_len();
	}

	auto matrix(const Index len_s, const Index len_t) {
		return m_factory.make(len_s, len_t);
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
};

template<typename Locality, typename Index=int16_t, typename Value=float>
class AffineGapCostAligner : public Aligner<Locality, Index, Value> {
	const Value m_gap_cost_s;
	const Value m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef Value GapCostSpec;

	inline AffineGapCostAligner(
		const Locality &p_locality,
		const Value p_gap_cost_s,
		const Value p_gap_cost_t,
		const Index p_max_len_s,
		const Index p_max_len_t) :

		Aligner<Locality, Index, Value>(p_locality, p_max_len_s, p_max_len_t),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		p_locality.init_border(
			this->m_factory.values(),
			xt::arange<Index>(0, p_max_len_s + 1) * p_gap_cost_s,
			xt::arange<Index>(0, p_max_len_t + 1) * p_gap_cost_t);
	}

	inline Value gap_cost_s(const size_t len) const {
		return m_gap_cost_s * len;
	}

	inline Value gap_cost_t(const size_t len) const {
		return m_gap_cost_t * len;
	}

	template<typename Flow, typename Similarity>
	Value compute(
		Flow &flow,
		const Similarity &similarity,
		const Index len_s,
		const Index len_t) const {

		// For global alignment, we pose the problem as a Needleman-Wunsch problem, but follow the
		// implementation of Sankoff and Kruskal.

		// For local alignments, we modify the problem for local  by adding a fourth zero case and
		// modifying the traceback (see Aluru or Hendrix).

		// The original paper by Smith and Waterman seems to contain an error in formula (3).

		// Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable
		// to the search for similarities in the amino acid sequence of two proteins.
		// Journal of Molecular Biology, 48(3), 443–453. https://doi.org/10.1016/0022-2836(70)90057-4

		// Smith, T. F., & Waterman, M. S. (1981). Identification of common
		// molecular subsequences. Journal of Molecular Biology, 147(1), 195–197.
		// https://doi.org/10.1016/0022-2836(81)90087-5

		// Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
		// the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

		// Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
		// String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045

		// Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
		// Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275

		// Hendrix, D. A. Applied Bioinformatics. https://open.oregonstate.education/appliedbioinformatics/.

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		auto matrix = this->m_factory.make(len_s, len_t);

		auto values = matrix.values();
		auto traceback = matrix.traceback();

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				MaxFold<Index, Value> best;

				best.set(
					values(u - 1, v - 1) + similarity(u, v),
					u - 1, v - 1);

				best.update(
					values(u - 1, v) - this->m_gap_cost_s,
					u - 1, v);

				best.update(
					values(u, v - 1) - this->m_gap_cost_t,
					u, v - 1);

				this->m_locality.update_best(best);

				values(u, v) = best.score();
				xt::view(traceback, u, v, xt::all()) = best.traceback();
			}
		}

		return this->m_locality.traceback(flow, matrix);
	}
};

template<typename Value>
inline void check_gap_tensor_shape(const xt::xtensor<Value, 1> &tensor, const size_t expected_len) {
	if (tensor.shape(0) != expected_len) {
		std::stringstream s;
		s << "expected gap cost tensor length of " << expected_len << ", got " << tensor.shape(0);
		throw std::invalid_argument(s.str());
	}
}

template<typename Locality, typename Index=int16_t, typename Value=float>
class GeneralGapCostAligner : public Aligner<Locality, Index, Value> {
	const xt::xtensor<Value, 1> m_gap_cost_s;
	const xt::xtensor<Value, 1> m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef GapTensorFactory GapCostSpec;

	inline GeneralGapCostAligner(
		const Locality &p_locality,
		const GapTensorFactory &p_gap_cost_s,
		const GapTensorFactory &p_gap_cost_t,
		const Index p_max_len_s,
		const Index p_max_len_t) :

		Aligner<Locality, Index, Value>(p_locality, p_max_len_s, p_max_len_t),
		m_gap_cost_s(p_gap_cost_s(p_max_len_s + 1)),
		m_gap_cost_t(p_gap_cost_t(p_max_len_t + 1)) {

		check_gap_tensor_shape(m_gap_cost_s, p_max_len_s + 1);
		check_gap_tensor_shape(m_gap_cost_t, p_max_len_t + 1);

		p_locality.init_border(
			this->m_factory.values(),
			m_gap_cost_s,
			m_gap_cost_t);
	}

	inline Value gap_cost_s(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost_s.shape(0));
		return m_gap_cost_s(len);
	}

	inline Value gap_cost_t(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost_t.shape(0));
		return m_gap_cost_t(len);
	}

	template<typename Flow, typename Similarity>
	Value compute(
		Flow &flow,
		const Similarity &similarity,
		const Index len_s,
		const Index len_t) const {

		// Our implementation follows what is commonly referred to as Waterman-Smith-Beyer, i.e.
		// an O(n^3) algorithm for generic gap costs. Waterman-Smith-Beyer generates a local alignment.

		// We use the same implementation approach as in the "affine_gap" method to differentiate
		// between local and global alignments.

		// Waterman, M. S., Smith, T. F., & Beyer, W. A. (1976). Some biological sequence metrics.
		// Advances in Mathematics, 20(3), 367–387. https://doi.org/10.1016/0001-8708(76)90202-4

		// Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
		// Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275

		// Hendrix, D. A. Applied Bioinformatics. https://open.oregonstate.education/appliedbioinformatics/.

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("length in general_gap must be >= 1");
		}

		auto matrix = this->m_factory.make(len_s, len_t);

		auto values = matrix.values();
		auto traceback = matrix.traceback();

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				MaxFold<Index, Value> best;

				best.set(
					values(u - 1, v - 1) + similarity(u, v),
					u - 1, v - 1);

				for (Index k = -1; k < u; k++) {
					best.update(
						values(k, v) - this->m_gap_cost_s(u - k),
						k, v);
				}

				for (Index k = -1; k < v; k++) {
					best.update(
						values(u, k) - this->m_gap_cost_t(v - k),
						u, k);
				}

				this->m_locality.update_best(best);

				values(u, v) = best.score();
				xt::view(traceback, u, v, xt::all()) = best.traceback();
			}
		}

		return this->m_locality.traceback(flow, matrix);
	}
};

} // namespace alignments

#endif // __VECTORIAN_ALIGNER__
