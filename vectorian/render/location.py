import collections
import roman


Location = collections.namedtuple("Location", ["speaker", "location"])


class PlayLocationFormatter:
	def __call__(self, document, location):
		speaker = location.get("speaker", 0)
		if speaker > 0:  # we have an act-scene-speakers structure.
			metadata = document.metadata
			act = location.get("act", 0)
			scene = location.get("scene", 0)
			line = location.get("line", 0)

			speaker = metadata["speakers"].get(speaker, "")
			if act > 0:
				return Location(speaker, "%s.%d, line %d" % (roman.toRoman(act), scene, line))
			else:
				return Location(speaker, "line %d" % line)
		else:
			return None


class BookLocationFormatter:
	def __call__(self, document, location):
		chapter = location.get("chapter", 0)

		if chapter > 0:  # book, chapter and paragraphs
			book = location.get("book", 0)
			chapter = location.get("chapter", 0)
			paragraph = location.get("paragraph", 0)

			if book <= 0:  # do we have a book?
				return Location("", "Chapter %d, par. %d" % (chapter, paragraph))
			else:
				return Location("", "Book %d, Chapter %d, par. %d" % (
					book, chapter, paragraph))
		else:
			return None


class TextLocationFormatter:
	def __call__(self, document, location):
		paragraph = location.get("paragraph", 0)
		if paragraph > 0:
			return Location("", "par. %d" % paragraph)
		else:
			return None


class LocationFormatter:
	def __init__(self):
		self._formatters = [
			PlayLocationFormatter(),
			BookLocationFormatter(),
			TextLocationFormatter()]

	def add(self, formatter):
		self._formatters.append(formatter)

	def __call__(self, document, location):
		for f in self._formatters:
			x = f(document, location)
			# print(f, "returned", x, "on", location)
			if x:
				return x
		return None
