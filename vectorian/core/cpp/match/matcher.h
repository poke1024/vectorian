#ifndef __VECTORIAN_MATCHER_H__
#define __VECTORIAN_MATCHER_H__

#include "common.h"

class Matcher : public std::enable_shared_from_this<Matcher> {
protected:
	const QueryRef m_query;
	const DocumentRef m_document;
	const MetricRef m_metric;

public:
	inline Matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric) :

		m_query(p_query),
		m_document(p_document),
		m_metric(p_metric) {
	}

	inline const QueryRef &query() const {
		return m_query;
	}

	inline const DocumentRef &document() const {
		return m_document;
	}

	inline const MetricRef &metric() const {
		return m_metric;
	}

	virtual ~Matcher();

	virtual void initialize() = 0;

	virtual void match(const ResultSetRef &p_matches) = 0;

	virtual float gap_cost(size_t len) const = 0;
};

typedef std::shared_ptr<Matcher> MatcherRef;

#endif // __VECTORIAN_MATCHER_H__
