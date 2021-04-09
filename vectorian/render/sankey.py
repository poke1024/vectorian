import bokeh
import bokeh.embed
import holoviews as hv
import json
import string
import collections
import logging

from vectorian.render.utils import flow_edges

hv.extension('bokeh')


class FlowRenderer:
	def __init__(self, cutoff=0, width=400, height_per_node=60, node_padding=80, cmap='Pastel1'):
		self._flows = {}
		self._cutoff = cutoff
		self._width = width
		self._height_per_node = height_per_node
		self._node_padding = node_padding
		self._cmap = cmap

	def _flow_to_sankey(self, match, flow):
		nodes = []
		node_mapping = collections.defaultdict(dict)
		spans = {'s': match.doc_span, 't': match.query}

		def token(name, i):
			idx = node_mapping[name]
			k = idx.get(i)
			if k is not None:
				return k
			idx[i] = len(nodes)
			nodes.append(' %s [%d] ' % (spans[name][i].text, i))
			return idx[i]

		edges = [(token('t', t), token('s', s), f) for t, s, f in flow_edges(flow, self._cutoff)]

		if len(edges) < 1:
			logging.warning("no edges found")

		n = max(
			len(set(x[0] for x in edges)),
			len(set(x[1] for x in edges)))

		nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
		return hv.Sankey((edges, nodes)).opts(
			width=self._width,
			height=n * self._height_per_node,
			labels='label',
			label_position='inner',
			cmap=self._cmap,
			node_padding=self._node_padding,
			show_values=False)

	def _script_code(self, iframe_id, div_id, match):
		flow = match.flow

		if flow is None:
			return

		sankey = self._flow_to_sankey(match, flow)
		fig = hv.render(sankey, backend='bokeh')
		fig.toolbar.logo = None
		fig.toolbar_location = None
		fig_json = json.dumps(bokeh.embed.json_item(fig, div_id))

		code = string.Template('''
		$('#${div_id}').ready(function () {
			Bokeh.embed.embed_item(${fig_json}).then(function() {
				parent.document.getElementById("${iframe_id}").onload();
			});
		});
		''')

		return code.safe_substitute(iframe_id=iframe_id, div_id=div_id, fig_json=fig_json)

	def write_head(self, doc):
		doc, tag, text = doc.tagtext()

		with tag('script', src='https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js'):
			pass

		bokeh_src = (
			f'https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh.__version__}.min.js',
			f'https://cdn.bokeh.org/bokeh/release/bokeh-widgets-{bokeh.__version__}.min.js',
			f'https://cdn.bokeh.org/bokeh/release/bokeh-tables-{bokeh.__version__}.min.js'
		)

		for src in bokeh_src:
			with tag('script', src=src, crossorigin='anonymous'):
				pass

	def write_match(self, doc, match, fetch_id):
		div_id = fetch_id()
		doc, tag, text = doc.tagtext()
		with tag('div', id=div_id):
			self._flows[div_id] = match

	def write_script(self, doc, iframe_id):
		doc, tag, text = doc.tagtext()
		with tag('script'):
			for div_id, match in self._flows.items():
				text(self._script_code(iframe_id, div_id, match))
