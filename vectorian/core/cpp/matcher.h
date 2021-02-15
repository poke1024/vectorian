#ifndef __VECTORIAN_MATCHER_H__
#define __VECTORIAN_MATCHER_H__

#include "common.h"

class Matcher {
public:
	virtual ~Matcher() {
	}

	virtual void match(const ResultSetRef &p_matches) = 0;
};

typedef std::shared_ptr<Matcher> MatcherRef;

#endif // __VECTORIAN_MATCHER_H__