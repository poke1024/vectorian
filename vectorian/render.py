import math

from yattag import Doc

# https://www.yattag.org/
# https://bulma.io/documentation/elements/tag/


def score_color_class(score):
	if score <= 0.75:
		return "tag is-warning"
	elif score <= 0.25:
		return "tag is-danger"
	else:
		return "tag is-success"


def trim_regions(regions):
	return regions

'''
joinRegions : Region -> List Region -> (String -> String -> String) -> List Region
joinRegions left rest joinStr
  = case List.head rest of
    Just h ->
      if h.similarity * h.weight == 0 then
        trimRegionsHead ([{
          s = (joinStr left.s  h.s), mismatch_penalty = 0, t = "", similarity = 0, weight = 0,
          pos_s = "", pos_t = "", metric = ""}] ++ (List.drop 1 rest)) joinStr
      else [left] ++ rest
    Nothing -> [left] ++ rest

trimRegionsHead : List Region -> (String -> String -> String) -> List Region
trimRegionsHead regions joinStr
  = case List.head regions of
    Just r ->
      if r.similarity * r.weight == 0
      then joinRegions r (List.drop 1 regions) joinStr
      else regions
    Nothing -> regions

trimRegions : List Region -> List Region
trimRegions r
  = List.reverse (trimRegionsHead (List.reverse (trimRegionsHead r (\a b -> a ++ b))) (\a b -> b ++ a))

'''

class Renderer:
	def __init__(self):
		doc, tag, text = Doc().tagtext()
		doc.asis('<!DOCTYPE html>')
		self._html = (doc, tag, text)

	def add_bold(self, s):
		doc, tag, text = self._html
		with tag('span', klass="has-text-black has-text-weight-bold"):
			text(s)

	def add_light(self, s):
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
						self.add_light(region['t'])
					text(" ")
					with tag('span', style=f'display:table-cell; opacity:{region["weight"]};'):
						with tag('span', klass=score_color_class(region['similarity'])):
							text("%d%%" % int(math.floor(100 * region['similarity'])))

		# FIXME annotate POS here.

	def add_region(self, region):
		if len(region.get('t', '')) > 0 and region['similarity'] * region['weight'] > 0:
			self.add_match_region(region)
		else:
			# FIXME annotate POS here.
			self.add_light(region['s'])

	def add_match_score(self, match):
		doc, tag, text = self._html
		with tag('span', klass='has-text-weight-bold'):
			text("%.1f%%" % (100 * match['score']))

	def add_match(self, match):
		doc, tag, text = self._html
		with tag('media'):
			with tag('mediaLeft'):
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

			with tag('mediaContent'):
				with tag('span', style='font-variant: small-caps;'):
					text(match['location']['speaker'])

				with tag('div', klass='is-pulled-right'):
					with tag('small'):
						text(match['location']['author'] + ', ')
					with tag('small', klass='is-italic'):
						text(match['location']['title'] + ', ')
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
		<div class="container">
			<div class="section">
'''
		epilog = '''
			</div>
		</div>
	</body>
</html>'''

		return ''.join([prolog, doc.getvalue(), epilog])
