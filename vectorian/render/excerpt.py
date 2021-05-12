import html
import math


def score_color_class(score):
	if score <= 0.75:
		return "tag is-warning"
	elif score <= 0.25:
		return "tag is-danger"
	else:
		return "tag is-success"


def trim_regions(regions):
	return regions


class ExcerptRenderer:
	def __init__(self, *args, context_size=10):
		# annotate: tags, metric, penalties
		self._context_size = context_size
		self._annotate = dict((x, True) for x in args)

	def add_context_text(self, doc, s):
		doc, tag, text = doc.tagtext()
		r = "&crarr;".join([html.escape(x) for x in s.split("\n")])
		doc.asis(r)

	def add_bold_text(self, doc, s):
		doc, tag, text = doc.tagtext()
		with tag('span', klass="has-text-black has-text-weight-bold"):
			self.add_context_text(doc, s)

	def add_light_text(self, doc, s):
		doc, tag, text = doc.tagtext()
		with tag('span', klass="has-text-grey-light"):
			self.add_context_text(doc, s)

	def add_light_tag(self, doc, s):
		doc, tag, text = doc.tagtext()
		with tag('span', klass="tag is-light"):
			text(s)

	def add_match_region(self, doc, region):
		doc, tag, text = doc.tagtext()
		edge = region['edges'][0]  # FIXME
		with tag('span'):
			with tag('span', style='display:inline-table;'):
				with tag('span', style='display:table-row;'):
					with tag('span', style='display:table-cell;'):
						self.add_bold_text(doc, region['s'])
						text(" ")
					with tag('span', style='display:table-cell;'):
						self.add_light_tag(doc, edge['t']['text'])
						text(" ")
					opacity = 0.5 + 0.5 * edge["flow"]
					with tag('span', style=f'display:table-cell; opacity:{opacity};'):
						similarity = 1 - edge["distance"]
						with tag('span', klass=score_color_class(similarity)):
							text("%d%%" % int(math.floor(100 * similarity)))

				if self._annotate.get('tags') or self._annotate.get('metric'):
					cell_style = 'display:table-cell; padding-left: 0.2em; padding-right: 0.2em;'

					with tag('span', style='display:table-row;'):
						if region['pos_s'] == edge['t']['pos']:
							text_class = 'has-text-black'
						else:
							text_class = 'has-text-danger'

						if self._annotate.get('tags'):
							with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered {text_class}'):
								if self._annotate.get('tags'):
									text(region['pos_s'])
							with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered'):
								if self._annotate.get('tags'):
									text(edge['t']['pos'])
						if self._annotate.get('metric'):
							with tag('span', style=cell_style, klass=f'is-size-7 has-text-centered has-text-grey-light'):
								if self._annotate.get('metric'):
									text(edge['metric'])

	def add_region(self, doc, region):
		if len(region.get('edges', [])) > 0:
			self.add_match_region(doc, region)
		elif self._annotate.get('penalties'):
			doc, tag, text = doc.tagtext()
			with tag('span', style='display:inline-table;'):
				with tag('span', style='display:table-row;'):
					self.add_light_text(doc, region['s'])
				with tag('span', klass='tag is-danger', style='display:table-row;'):
					if region['gap_penalty'] > 0:
						text('-%.1f' % (region['gap_penalty'] * 100))
					else:
						text('')
		else:
			self.add_light_text(doc, region['s'])

	def write_head(self, doc):
		pass

	def write_match(self, doc, match_obj, fetch_id):
		doc, tag, text = doc.tagtext()

		match = match_obj.to_json(
			self._context_size)

		with tag('span'):
			regions = trim_regions(match['regions'])
			if match['level'] == 'span':
				for i, r in enumerate(regions):
					text(r['s'])
					if i < len(regions) - 1:
						text(" ")
			else:
				for i, r in enumerate(regions):
					self.add_region(doc, r)
					if i < len(regions) - 1:
						text(" ")

	def write_script(self, doc, iframe_id):
		pass
