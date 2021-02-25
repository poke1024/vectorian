import math
import html
import time
import string

from yattag import Doc


def score_color_class(score):
	if score <= 0.75:
		return "tag is-warning"
	elif score <= 0.25:
		return "tag is-danger"
	else:
		return "tag is-success"


def trim_regions(regions):
	return regions


class Renderer:
	def __init__(self, annotate=None):
		doc, tag, text = Doc().tagtext()
		doc.asis('<!DOCTYPE html>')
		self._html = (doc, tag, text)
		self._annotate = annotate or {}

	def add_bold(self, s):
		doc, tag, text = self._html
		with tag('span', klass="has-text-black has-text-weight-bold"):
			text(s)

	def add_light_tag(self, s):
		doc, tag, text = self._html
		with tag('span', klass="tag is-light"):
			text(s)

	def add_light_text(self, s):
		doc, tag, text = self._html
		with tag('span', klass="has-text-grey-light"):
			text(s)

	def add_match_region(self, region):
		doc, tag, text = self._html
		with tag('span'):
			with tag('span', style='display:inline-table;'):
				with tag('span', style='display:table-row;'):
					with tag('span', style='display:table-cell;'):
						self.add_bold(region['s'])
					text(" ")
					with tag('span', style='display:table-cell;'):
						self.add_light_tag(region['t'])
					text(" ")
					with tag('span', style=f'display:table-cell; opacity:{region["weight"]};'):
						with tag('span', klass=score_color_class(region['similarity'])):
							text("%d%%" % int(math.floor(100 * region['similarity'])))

				if self._annotate.get('tags') or self._annotate.get('metric'):
					cell_style = 'display:table-cell; padding-left: 0.2em; padding-right: 0.2em;'

					with tag('span', style='display:table-row;'):
						if region['pos_s'] == region['pos_t']:
							text_class = 'has-text-black'
						else:
							text_class = 'has-text-danger'

						with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered {text_class}'):
							if self._annotate.get('tags'):
								text(region['pos_s'])
						with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered'):
							if self._annotate.get('tags'):
								text(region['pos_t'])
						with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered has-text-grey-light'):
							if self._annotate.get('metric'):
								text(region['metric'])

	def add_region(self, region):
		if len(region.get('t', '')) > 0 and region['similarity'] * region['weight'] > 0:
			self.add_match_region(region)
		else:
			# FIXME annotate POS here.
			self.add_light_text(region['s'])

	def add_match_score(self, match):
		doc, tag, text = self._html
		with tag('span', klass='has-text-weight-bold'):
			text("%.1f%%" % (100 * match['score']))

	def add_match(self, match):
		doc, tag, text = self._html
		with tag('article', klass="media"):
			with tag('div', klass='media-left'):
				with tag('p', klass='image is-64x64'):
					with tag('span', klass='buttons'):
						self.add_match_score(match)
						doc.stag('br')
						with tag('div'):
							if len(match['omitted']) <= 2:
								for x in match['omitted']:
									with tag('div', style='text-decoration: line-through;'):
										text(x)
							else:
								with tag('div', style='white-space: nowrap;'):
									text(f"{len(match['omitted'])} omitted")

					# FIXME annotateDebug

			with tag('div', klass='media-content'):
				speaker = match['location']['speaker']
				title = match['location']['title']

				if speaker:
					with tag('span', style='font-variant: small-caps;'):
						text(match['location']['speaker'])

				with tag('div', klass='is-pulled-right'):
					if speaker:
						with tag('small'):
							text(match['location']['author'] + ', ')
					if title:
						with tag('small', klass='is-italic'):
							text(title + ', ')
					with tag('small'):
						text(match['location']['location'])

				with tag('div'):
					doc.stag('br')
					doc.stag('br')
					with tag('span'):
						regions = trim_regions(match['regions'])
						for i, r in enumerate(regions):
							self.add_region(r)
							if i < len(regions) - 1:
								text(" ")

	def to_html(self):
		doc, tag, text = self._html

		prolog = '''<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Vectorian</title>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.1/css/bulma.min.css">
	</head>
	<body>
		<div class="container" height="100%">
			<div class="section">
'''
		epilog = '''
			</div>
		</div>
	</body>
</html>'''

		# see https://github.com/ipython/ipython/blob/master/IPython/lib/display.py

		iframe_id = f"vectorian-iframe-{time.time_ns()}"

		iframe = string.Template("""
		        <iframe
		        	id="$id"
		            width="$width"
		            height="$height"
		            srcdoc="$srcdoc"
		            frameborder="0"
		            allowfullscreen
		        ></iframe>
		        <script>
		        	var f = document.getElementById("$id");
		        	f.onload = function() {
			        	f.height = f.contentWindow.document.body.scrollHeight + "px";
			        };
		        </script>
		""")

		s = ''.join([prolog, doc.getvalue(), epilog])
		return iframe.safe_substitute(dict(
			id=iframe_id, width="100%", height="100%", srcdoc=html.escape(s)))
