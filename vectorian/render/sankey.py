import bokeh
import bokeh.embed
import holoviews as hv
import json
import string
import collections
import logging

hv.extension('bokeh')


def flow_to_sankey(match, flow, cutoff=0.1):
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

	edges = []

	if flow['type'] == 'injective':

		for t, (s, f) in enumerate(zip(flow['target'], flow['flow'])):
			if s >= 0 and f > cutoff:
				edges.append((token('t', t), token('s', s), f))

	elif flow['type'] == 'sparse':

		for t, s, f in zip(flow['source'], flow['target'], flow['flow']):
			if f > cutoff:
				edges.append((token('t', t), token('s', s), f))

	elif flow['type'] == 'dense':

		m = flow['flow']
		for t in range(m.shape[0]):
			for s in range(m.shape[1]):
				f = m[t, s]
				if f > cutoff:
					edges.append((token('t', t), token('s', s), f))

	else:
		raise ValueError(flow['type'])

	if len(edges) < 1:
		logging.warning("no edges found")

	n = max(
		len(set(x[0] for x in edges)),
		len(set(x[1] for x in edges)))

	nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
	return hv.Sankey((edges, nodes)).opts(
		width=400,
		height=n * 60,
		labels='label',
		label_position='inner',
		cmap='Pastel1',
		node_padding=80,
		show_values=False)


def script_code(iframe_id, div_id, match):
	flow = match.flow

	if flow is None:
		return

	sankey = flow_to_sankey(match, flow)
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


class FlowRenderer:
	def __init__(self):
		self._flows = {}

	def write_match(self, doc, match, fetch_id):
		div_id = fetch_id()
		doc, tag, text = doc.tagtext()
		with tag('div', id=div_id):
			self._flows[div_id] = match

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

	def write_script(self, doc, iframe_id):
		doc, tag, text = doc.tagtext()
		with tag('script'):
			for div_id, match in self._flows.items():
				text(script_code(iframe_id, div_id, match))