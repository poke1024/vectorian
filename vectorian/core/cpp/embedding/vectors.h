#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

class ContextualVectorsContainer {
protected:
	std::unordered_map<std::string, py::object> m_contextual_vectors;

public:
	py::object get_contextual_embedding_vectors(
		const std::string &p_name) const {

		auto it = m_contextual_vectors.find(p_name);
		if (it == m_contextual_vectors.end()) {
			std::ostringstream err;
			err << "could not find embedding " << p_name;
			throw std::runtime_error(err.str());
		}
		return it->second;
	}

};

#endif // __VECTORIAN_WORD_VECTORS_H__
