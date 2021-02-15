#ifndef __VECTORIAN_MISMATCH_PENALTY_H__
#define __VECTORIAN_MISMATCH_PENALTY_H__

#include "common.h"

class MismatchPenalty {
	std::vector<float> m_penalties;

public:
	MismatchPenalty(float cutoff, int max_n) {
		m_penalties.resize(max_n);

		if (cutoff < 0.0f) { // disabled / off?
			for (int i = 0; i < max_n; i++) {
				m_penalties[i] = 0.0f;
			}
		} else {
			// this cumbersome formulation is equivalent to (1 - 2^-(i / cutoff))

			const float scale = cutoff / 0.693147;

			for (int i = 0; i < max_n; i++) {
				m_penalties[i] = std::min(1.0f, 1.0f - exp(-i / scale));
			}
		}
	}

	inline double operator()(const size_t x) const {
		if (x < m_penalties.size()) {
			return m_penalties.at(x);
		} else {
			return 1e10;
		}
	}
};

typedef std::shared_ptr<MismatchPenalty> MismatchPenaltyRef;

#endif // __VECTORIAN_MISMATCH_PENALTY_H__
