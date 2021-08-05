#ifndef __VECTORIAN_ALIGNER__
#define __VECTORIAN_ALIGNER__

#include "common.h"

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
	const size_t m_max_len_s;
	const size_t m_max_len_t;

	inline void check_size_against_max(const size_t p_len, const size_t p_max) const {
		if (p_len > p_max) {
			std::stringstream err;
			err << "sequence of length " << p_len << " exceeds configured maximum length " << p_max;
			throw std::invalid_argument(err.str());
		}
	}

	inline void check_size_against_implementation_limit(const size_t p_len) const {
		const size_t max = size_t(std::numeric_limits<Index>::max()) - 4;
		if (p_len > max) {
			std::stringstream err;
			err << "maximum supported sequence length in this implementation is " <<
				max << ", but given sequence has length " << p_len;
			throw std::invalid_argument(err.str());
		}
	}

public:
	inline MatrixFactory(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		m_data(std::make_unique<Data>()),
		m_max_len_s(p_max_len_s),
		m_max_len_t(p_max_len_t) {

		check_size_against_implementation_limit(p_max_len_s);
		check_size_against_implementation_limit(p_max_len_t);

		m_data->values.resize({
			m_max_len_s + 1,
			m_max_len_t + 1
		});
		m_data->traceback.resize({
			m_max_len_s,
			m_max_len_t,
			2
		});
		m_data->best_column.resize({
			m_max_len_s
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
		// a custom view to make sure that negative indexes, e.g. m(-1, 2), are handled correctly.

		auto &v = m_factory.m_data->values;

		return [&v] (const Index i, const Index j) -> typename xt::xtensor<Value, 2>::reference {
			return v(i + 1, j + 1);
		};
	}

	inline auto values_non_neg_ij() const {
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

	check_size_against_max(len_s, m_max_len_s);
	check_size_against_max(len_t, m_max_len_t);
	return Matrix(*this, len_s, len_t);
}

template<typename V>
inline size_t argmax(const V &v) {
	// we do not use xt::argmax here,
	// since we want a guaranteed behaviour
	// (lowest index) on ties.

	const size_t n = v.shape(0);

	auto best = v(0);
	size_t best_i = 0;

	for (size_t i = 1; i < n; i++) {
		const auto x = v(i);
		if (x > best) {
			best = x;
			best_i = i;
		}
	}

	return best_i;
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

	template<typename Alignment, typename Index>
	inline Value traceback(
		Alignment &alignment,
		Matrix<Index, Value> &matrix) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.values();
		const auto traceback = matrix.traceback();
		auto best_column = matrix.best_column();

		const auto zero_similarity = m_zero;

		best_column = xt::argmax(matrix.values_non_neg_ij(), 1);

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

		alignment.resize(len_s, len_t);

		Index u = best_u;
		Index v = best_v;

		while (u >= 0 && v >= 0 && values(u, v) > zero_similarity) {
			const Index last_u = u;
			const Index last_v = v;

			const auto t = xt::view(traceback, u, v, xt::all());
			u = t(0);
			v = t(1);

			if (u != last_u && v != last_v) {
				alignment.add_edge(last_u, last_v);
			}
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

	template<typename Alignment, typename Index>
	inline float traceback(
		Alignment &alignment,
		Matrix<Index, Value> &matrix) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.values();
		const auto traceback = matrix.traceback();

		alignment.resize(len_s, len_t);

		Index u = len_s - 1;
		Index v = len_t - 1;
		const Value best_score = values(u, v);

		while (u >= 0 && v >= 0) {
			const Index last_u = u;
			const Index last_v = v;

			const auto t = xt::view(traceback, u, v, xt::all());
			u = t(0);
			v = t(1);

			if (u != last_u && v != last_v) {
				alignment.add_edge(last_u, last_v);
			}
		}

		return best_score;
	}
};


template<typename Value>
class SemiGlobal {
public:
	void init_border(
		xt::xtensor<Value, 2> &p_values,
		const xt::xtensor<Value, 1> &p_gap_cost_s,
		const xt::xtensor<Value, 1> &p_gap_cost_t) const {

		xt::view(p_values, xt::all(), 0).fill(0);
		xt::view(p_values, 0, xt::all()).fill(0);
	}

	template<typename Fold>
	inline void update_best(Fold &fold) const {
	}

	template<typename Alignment, typename Index>
	inline float traceback(
		Alignment &alignment,
		Matrix<Index, Value> &matrix) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.values();
		const auto traceback = matrix.traceback();

		alignment.resize(len_s, len_t);

		const Index last_row = len_s - 1;
		const Index last_col = len_t - 1;

		const auto values_non_neg_ij = matrix.values_non_neg_ij();
		const Index best_col_in_last_row = argmax(xt::row(values_non_neg_ij, last_row));
		const Index best_row_in_last_col = argmax(xt::col(values_non_neg_ij, last_col));

		Index u;
		Index v;

		if (values(best_row_in_last_col, last_col) > values(last_row, best_col_in_last_row)) {
			u = best_row_in_last_col;
			v = last_col;
		} else {
			u = last_row;
			v = best_col_in_last_row;
		}

		const Value best_score = values(u, v);

		while (u >= 0 && v >= 0) {
			const Index last_u = u;
			const Index last_v = v;

			const auto t = xt::view(traceback, u, v, xt::all());
			u = t(0);
			v = t(1);

			if (u != last_u && v != last_v) {
				alignment.add_edge(last_u, last_v);
			}
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
class Solver {
protected:
	const Locality m_locality;
	MatrixFactory<Index, Value> m_factory;

public:
	inline Solver(
		const Locality &p_locality,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		m_locality(p_locality),
		m_factory(p_max_len_s, p_max_len_t) {
	}

	inline Index max_len() const {
		return m_factory.max_len();
	}

	auto matrix(const Index len_s, const Index len_t) {
		return m_factory.make(len_s, len_t);
	}
};

template<typename Locality, typename Index=int16_t, typename Value=float>
class AffineGapCostSolver : public Solver<Locality, Index, Value> {
	const Value m_gap_cost_s;
	const Value m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef Value GapCostSpec;

	inline AffineGapCostSolver(
		const Locality &p_locality,
		const Value p_gap_cost_s,
		const Value p_gap_cost_t,
		const Index p_max_len_s,
		const Index p_max_len_t) :

		Solver<Locality, Index, Value>(p_locality, p_max_len_s, p_max_len_t),
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

	template<typename Alignment, typename Similarity>
	Value solve(
		Alignment &alignment,
		const Similarity &similarity,
		const size_t len_s,
		const size_t len_t) const {

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

		auto matrix = this->m_factory.make(len_s, len_t);

		auto values = matrix.values();
		auto traceback = matrix.traceback();

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

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

		return this->m_locality.traceback(alignment, matrix);
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
class GeneralGapCostSolver : public Solver<Locality, Index, Value> {
	const xt::xtensor<Value, 1> m_gap_cost_s;
	const xt::xtensor<Value, 1> m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef GapTensorFactory GapCostSpec;

	inline GeneralGapCostSolver(
		const Locality &p_locality,
		const GapTensorFactory &p_gap_cost_s,
		const GapTensorFactory &p_gap_cost_t,
		const Index p_max_len_s,
		const Index p_max_len_t) :

		Solver<Locality, Index, Value>(p_locality, p_max_len_s, p_max_len_t),
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

	template<typename Alignment, typename Similarity>
	Value solve(
		Alignment &alignment,
		const Similarity &similarity,
		const size_t len_s,
		const size_t len_t) const {

		// Our implementation follows what is commonly referred to as Waterman-Smith-Beyer, i.e.
		// an O(n^3) algorithm for generic gap costs. Waterman-Smith-Beyer generates a local alignment.

		// We use the same implementation approach as in the "affine_gap" method to differentiate
		// between local and global alignments.

		// Waterman, M. S., Smith, T. F., & Beyer, W. A. (1976). Some biological sequence metrics.
		// Advances in Mathematics, 20(3), 367–387. https://doi.org/10.1016/0001-8708(76)90202-4

		// Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
		// Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275

		// Hendrix, D. A. Applied Bioinformatics. https://open.oregonstate.education/appliedbioinformatics/.

		auto matrix = this->m_factory.make(len_s, len_t);

		auto values = matrix.values();
		auto traceback = matrix.traceback();

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

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

		return this->m_locality.traceback(alignment, matrix);
	}
};

} // namespace alignments

#endif // __VECTORIAN_ALIGNER__