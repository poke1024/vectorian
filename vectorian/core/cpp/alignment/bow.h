#ifndef __VECTORIAN_BOW__
#define __VECTORIAN_BOW__

template<typename Index>
class BOWProblem {
public:
	typedef std::vector<Index> IndexVector;

	struct Document {
		std::vector<float> bow; // (n)bow
		Index w_sum;
		std::vector<Index> vocab;
		std::vector<Index> pos_to_vocab;
		std::vector<IndexVector> vocab_to_pos; // 1:n

		void allocate(const size_t p_size) {
			bow.resize(p_size);
			vocab.reserve(p_size);
			pos_to_vocab.resize(p_size);
			vocab_to_pos.reserve(p_size);
			for (size_t i = 0; i < p_size; i++) {
				vocab_to_pos.emplace_back(std::vector<Index>());
				vocab_to_pos.back().reserve(p_size);
			}
		}
	};

	Document m_doc[2]; // s, t
	size_t m_max_size; // i.e. pre-allocation
	size_t m_vocabulary_size;

	inline BOWProblem() : m_max_size(0), m_vocabulary_size(0) {
	}

	void allocate(
		const size_t max_len_s,
		const size_t max_len_t) {

		PPK_ASSERT(max_len_s > 0);
		PPK_ASSERT(max_len_t > 0);

		const size_t size = max_len_s + max_len_t;

		m_max_size = size;

		for (int i = 0; i < 2; i++) {
			m_doc[i].allocate(size);
		}
	}

	inline auto bow(const int p_doc) const {
		return xt::adapt(
			m_doc[p_doc].bow.data(),
			{m_vocabulary_size});
	}

	py::dict py_vocab_to_pos(const int p_doc) const {
		py::dict result;
		const auto &doc = m_doc[p_doc];
		for (size_t i = 0; i < m_vocabulary_size; i++) {
			const auto &mapping = doc.vocab_to_pos[i];
			if (!mapping.empty()) {
				py::list positions;
				for (auto p : mapping) {
					positions.append(p);
				}
				result[py::int_(i)] = positions;
			}
		}
		return result;
	}

	inline void reset(const int k) {
		for (int j = 0; j < 2; j++) {
			float *w = m_doc[j].bow.data();
			for (int i = 0; i < k; i++) {
				w[i] = 0.0f;
			}
		}
	}

	template<typename Slice, typename BuildBOW>
	inline bool initialize(
		const Slice &slice,
		BuildBOW &build_bow,
		const bool normalize_bow) {

		const auto vocab_size = build_bow.build(
			slice, *this, normalize_bow);
		if (vocab_size == 0) {
			return false;
		}

		m_vocabulary_size = vocab_size;
		return true;
	}
};

struct TaggedTokenId {
	token_t token;
	int8_t tag;

	inline bool operator==(const TaggedTokenId &t) const {
		return token == t.token && tag == t.tag;
	}

	inline bool operator!=(const TaggedTokenId &t) const {
		return !(*this == t);
	}

	inline bool operator<(const TaggedTokenId &t) const {
		if (token < t.token) {
			return true;
		} else if (token == t.token) {
			return tag < t.tag;
		} else {
			return false;
		}
	}
};

class UntaggedTokenFactory {
public:
	typedef size_t Token;

	template<typename Slice>
	Token s(
		const Slice &p_slice,
		const size_t i) const {

		return p_slice.encoder().to_embedding(0, i, p_slice.s(i));
	}

	template<typename Slice>
	Token t(
		const Slice &p_slice,
		const size_t i) const {

		return p_slice.encoder().to_embedding(1, i, p_slice.t(i));
	}
};

class TaggedTokenFactory {
public:
	typedef TaggedTokenId Token;

	template<typename Slice>
	Token s(
		const Slice &p_slice,
		const size_t i) const {

		const auto &token = p_slice.s(i);
		return Token{
			p_slice.encoder().to_embedding(0, i, token),
			token.tag
		};
	}

	template<typename Slice>
	Token t(
		const Slice &p_slice,
		const size_t i) const {

		const auto &token = p_slice.t(i);
		return Token{
			p_slice.encoder().to_embedding(1, i, token),
			token.tag
		};
	}
};

template<typename Index, typename TokenFactory>
class BOWBuilder {
public:
	typedef typename TokenFactory::Token Token;

	struct RefToken {
		Token id; // unique id for token
		Index pos; // index in s or t
		int8_t doc; // 0 for s, 1 for t
	};

	const TokenFactory m_token_factory;
	std::vector<RefToken> m_tokens;

	inline BOWBuilder() {
	}

	inline BOWBuilder(const TokenFactory &p_token_factory) :
		m_token_factory(p_token_factory) {
	}

	inline void allocate(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_tokens.resize(max_len_s + max_len_t);
	}

	template<typename Slice>
	size_t build(
		const Slice &p_slice,
		BOWProblem<Index> &p_problem,
		const bool p_normalize_bow)  {

		const auto len_s = p_slice.len_s();
		const auto len_t = p_slice.len_t();

		Index k = 0;
		std::vector<RefToken> &z = m_tokens;

		for (Index i = 0; i < len_s; i++) {
			z[k++] = RefToken{
				m_token_factory.s(p_slice, i), i, 0};
		}
		for (Index i = 0; i < len_t; i++) {
			z[k++] = RefToken{
				m_token_factory.t(p_slice, i), i, 1};
		}

		if (k == 0) {
			return 0;
		}

		std::sort(z.begin(), z.begin() + k, [] (const RefToken &a, const RefToken &b) {
			return a.id < b.id;
		});

		p_problem.reset(k);

		for (int i = 0; i < 2; i++) {
			auto &doc = p_problem.m_doc[i];
			doc.w_sum = 0;
			doc.vocab.clear();
			doc.vocab_to_pos[0].clear();
		}

		auto cur_word_id = m_tokens[0].id;
		Index vocab = 0;

		for (Index i = 0; i < k; i++) {
			const auto &token = m_tokens[i];
			const auto new_word_id = token.id;

			if (new_word_id != cur_word_id) {
				cur_word_id = new_word_id;
				vocab += 1;
				for (int j = 0; j < 2; j++) {
					p_problem.m_doc[j].vocab_to_pos[vocab].clear();
				}
			}

			const int doc_idx = token.doc;
			auto &doc = p_problem.m_doc[doc_idx];

			doc.bow[vocab] += 1.0f;
			doc.w_sum += 1;
			doc.pos_to_vocab[token.pos] = vocab;

			auto &to_pos = doc.vocab_to_pos[vocab];
			if (to_pos.empty()) {
				doc.vocab.push_back(vocab);
			}
			to_pos.push_back(token.pos);
		}

		if (p_normalize_bow) {
			for (int c = 0; c < 2; c++) {
				float *w = p_problem.m_doc[c].bow.data();
				const float s = p_problem.m_doc[c].w_sum;
				for (const Index i : p_problem.m_doc[c].vocab) {
					w[i] /= s;
				}
			}
		}

		return vocab + 1;
	}
};

#endif // __VECTORIAN_BOW__
