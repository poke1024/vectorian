import html
import time
import string

from yattag import Doc
from vectorian.index import PartitionData


class Renderer:
	# see https://github.com/ipython/ipython/blob/master/IPython/lib/display.py
	_iframe_template = string.Template('''
	<iframe
		id="$id"
		width="$width"
		height="$height"
		srcdoc="$srcdoc"
		onload="$onload"
		frameborder="0"
		allowfullscreen
	></iframe>
	''')

	_onload = string.Template("""
		(function() {
			var f = parent.document.getElementById('${iframe_id}');
			f.height = f.contentWindow.document.body.scrollHeight + 'px';
		})();
	""")

	def __init__(self, renderers, location_formatter, annotate=None):
		self._annotate = annotate or {}

		self._renderers = renderers
		self._location_formatter = location_formatter

		self._id_base = f"vectorian-{time.time_ns()}-{id(self)}"
		self._id_index = 0

	def _next_unique_id(self):
		self._id_index += 1
		return self._id_base + f"-{self._id_index}"

	def add_match_score(self, doc, match):
		doc, tag, text = doc.tagtext()
		with tag('span', klass='has-text-weight-bold'):
			text("%.1f%%" % (100 * match.score))

	def add_match(self, doc, match):
		doc, tag, text = doc.tagtext()

		with tag('article', klass="media"):
			with tag('div', klass='media-left'):
				with tag('p', klass='image is-64x64'):
					with tag('span', klass='buttons'):
						self.add_match_score(doc, match)

						omitted = match.omitted
						if omitted:
							doc.stag('br')
							with tag('div'):
								if len(omitted) <= 2:
									for x in omitted:
										with tag('div', style='text-decoration: line-through;'):
											text(x)
								else:
									with tag('div', style='white-space: nowrap;'):
										text(f"{len(omitted)} omitted")

						if self._annotate.get('metadata'):
							doc.stag('br')
							with tag('div', klass="is-size-7"):
								for k, v in match.document.metadata.items():
									if v:
										with tag('div'):
											text(f"doc/{k}: {v}")
								with tag('div'):
									text("slice: " + str(match.slice_id))

			match_doc = match.prepared_doc

			partition = match.query.options["partition"]
			span_info = match_doc.span_info(
				PartitionData(**partition), match.slice_id)

			metadata = match_doc.metadata
			loc = self._location_formatter(match_doc, span_info)
			if loc:
				speaker, loc_desc = loc
			else:
				speaker = ""
				loc_desc = ""
			r_location = dict(
				speaker=speaker,
				author=metadata["author"],
				title=metadata["title"],
				location=loc_desc)

			with tag('div', klass='media-content'):
				speaker = r_location['speaker']
				title = r_location['title']

				if speaker:
					with tag('span', style='font-variant: small-caps;'):
						text(r_location['speaker'])

				with tag('div', klass='is-pulled-right'):
					appended = False
					if r_location['author']:
						with tag('small'):
							text(r_location['author'])
							appended = True
					if title:
						with tag('small', klass='is-italic'):
							if appended:
								text(', ')
							text(title)
							appended = True
					if r_location['location']:
						with tag('small'):
							if appended:
								text(', ')
							text(r_location['location'])

				with tag('div'):
					doc.stag('br')
					doc.stag('br')

					for renderer in self._renderers:
						renderer.write_match(doc, match, self._next_unique_id)

	def to_bare_bones_html(self, matches):
		doc, tag, text = Doc().tagtext()

		for match_obj in matches:
			self.add_match(doc, match_obj)

		return doc.getvalue()

	def to_html(self, matches):
		doc, tag, text = Doc().tagtext()
		iframe_id = self._next_unique_id()

		doc.asis('<!DOCTYPE html>')
		with tag('html'):
			with tag('head'):
				doc.asis('<meta charset="utf-8">')
				doc.asis('<meta name="viewport" content="width=device-width, initial-scale=1">')

				doc.stag(
					'link', rel='stylesheet',
					href='https://cdn.jsdelivr.net/npm/bulma@0.9.1/css/bulma.min.css')

				for renderer in self._renderers:
					renderer.write_head(doc)

			with tag('body'):
				with tag('div', klass='container', height='100%'):
					with tag('div', klass='section'):
						for match_obj in matches:
							self.add_match(doc, match_obj)

				for renderer in self._renderers:
					renderer.write_script(doc, iframe_id)

		# we need the following script_code for widget outputs to
		# work (correctly resize) in Jupyter Lab interactive mode.

		onload = Renderer._onload.safe_substitute(iframe_id=iframe_id)
		script_code = ''.join(['<script>', onload, '</script>'])

		return Renderer._iframe_template.safe_substitute(
			id=iframe_id, width="100%", height="100%",
			srcdoc=html.escape(doc.getvalue() + script_code),
			onload=onload)
