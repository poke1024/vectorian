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

	virtual float gap_cost_s(size_t len) const = 0;
	virtual float gap_cost_t(size_t len) const = 0;
};

typedef std::shared_ptr<Matcher> MatcherRef;

class MatcherFactory;
typedef std::shared_ptr<MatcherFactory> MatcherFactoryRef;

template<typename Make>
class MatcherFactoryImpl;

class MinimalMatcherFactory {
protected:
	typedef int16_t Index;

	template<typename GenSlices>
	MatcherRef make_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document,
		const MatcherOptions &p_matcher_options,
		const GenSlices &p_gen_slices) const;

public:
	virtual ~MinimalMatcherFactory() {
	}

    virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document,
		const MatcherOptions &p_matcher_options) const = 0;
};

typedef std::shared_ptr<MinimalMatcherFactory> MinimalMatcherFactoryRef;

class MatcherFactory {
	const MinimalMatcherFactoryRef m_static_factory;
	const MinimalMatcherFactoryRef m_contextual_factory;
	const MatcherOptions m_options;

public:
	inline MatcherFactory(
		const MinimalMatcherFactoryRef &p_static_factory,
		const MinimalMatcherFactoryRef &p_contextual_factory,
		const MatcherOptions &p_options) :

		m_static_factory(p_static_factory),
		m_contextual_factory(p_contextual_factory),
		m_options(p_options) {
	}

	MatcherRef create_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document) const;

	inline const MatcherOptions &options() const {
		return m_options;
	}

	inline bool needs_magnitudes() const {
		return m_options.needs_magnitudes;
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

	virtual float gap_cost_s(size_t len) const {
		return 0.0f;
	}

	virtual float gap_cost_t(size_t len) const {
		return 0.0f;
	}
};

typedef std::shared_ptr<ExternalMatcher> ExternalMatcherRef;

#endif // __VECTORIAN_MATCHER_H__
