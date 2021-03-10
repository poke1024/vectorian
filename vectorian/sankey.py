import bokeh
import bokeh.embed
import holoviews as hv
import json
import string
import collections

hv.extension('bokeh')


def write_head(tag):
	bokeh_src = (
		f'https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh.__version__}.min.js',
		f'https://cdn.bokeh.org/bokeh/release/bokeh-widgets-{bokeh.__version__}.min.js',
		f'https://cdn.bokeh.org/bokeh/release/bokeh-tables-{bokeh.__version__}.min.js'
	)

	for src in bokeh_src:
		with tag('script', src=src, crossorigin='anonymous'):
			pass


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

		for t, (s, w) in enumerate(zip(flow['target'], flow['weight'])):
			if s >= 0 and w > cutoff:
				edges.append((token('t', t), token('s', s), w))

	elif flow['type'] == 'sparse':

		for t, s, w in zip(flow['source'], flow['target'], flow['weight']):
			if w > cutoff:
				edges.append((token('t', t), token('s', s), w))

	else:
		raise ValueError(flow['type'])

	n = max(
		len(set(x[0] for x in edges)),
		len(set(x[1] for x in edges)))

	nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
	return hv.Sankey((edges, nodes)).opts(
		width=900,
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
