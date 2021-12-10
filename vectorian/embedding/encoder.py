class EmbeddingEncoder:
    @property
    def is_static(self):
        return False

    @property
    def is_contextual(self):
        return False

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def embedding(self):
        raise NotImplementedError()

    @property
    def dimension(self):
        raise NotImplementedError()

    def encode(self, docs):
        raise NotImplementedError()
