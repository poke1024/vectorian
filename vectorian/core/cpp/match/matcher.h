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

class MatcherFactory;
typedef std::shared_ptr<MatcherFactory> MatcherFactoryRef;

template<typename Make>
class MatcherFactoryImpl;

class MatcherFactory {
public:
	virtual MatcherRef create_matcher(const DocumentRef &p_document) const = 0;

	virtual ~MatcherFactory() {
	}

	template<typename Make>
	static inline MatcherFactoryRef create(const Make &p_make) {
		return std::make_shared<MatcherFactoryImpl<Make>>(p_make);
	}
};

template<typename Make>
class MatcherFactoryImpl : public MatcherFactory {
	const Make m_make;

public:
    inline MatcherFactoryImpl(const Make &p_make) : m_make(p_make) {
	}

	virtual MatcherRef create_matcher(const DocumentRef &p_document) const {
		return m_make(p_document);
	}
};

class ExternalMatcher : public Matcher {
public:
	ExternalMatcher(const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric) : Matcher(p_query, p_document, p_metric) {
	}

	virtual void initialize() {
		throw std::runtime_error(
			"ExternalMatcher::initialize is not allowed");
	}

	virtual void match(const ResultSetRef &p_matches) {
		throw std::runtime_error(
			"ExternalMatcher::match is not allowed");
	}

	virtual float gap_cost(size_t len) const {
		return 0.0f;
	}
};

typedef std::shared_ptr<ExternalMatcher> ExternalMatcherRef;

#endif // __VECTORIAN_MATCHER_H__
