import string
import os
import json

from pathlib import Path
from vectorian.render.vega import VegaRenderer
from vectorian.render.utils import flow_edges


class MatrixRenderer(VegaRenderer):
	_template = None

	def __init__(self):
		super().__init__()
		if MatrixRenderer._template is None:
			script_dir = Path(os.path.realpath(__file__)).parent
			with open(script_dir / 'vega' / 'matrix.json', 'r') as f:
				MatrixRenderer._template = string.Template(f.read())

	def _make_data(self, rows, columns, edges):
		source_nodes = []
		for i, token in enumerate(rows):
			source_nodes.append({
				'name': token.text,
				'group': 1,
				'index': len(source_nodes)
			})

		target_nodes = []
		for i, token in enumerate(columns):
			target_nodes.append({
				'name': token.text,
				'group': 1,
				'index': len(target_nodes)
			})

		links = []
		for i, j, w in edges:
			if w > 0:
				links.append({
					'source': i,
					'target': j,
					'value': w
				})

		return {
			'source_nodes': source_nodes,
			'target_nodes': target_nodes,
			'links': links}

	def _make_vega_spec(self, match):
		flow = match.flow

		if flow is None:
			return ''

		spans = {'s': match.doc_span, 't': match.query}

		data = self._make_data(
			list(spans['t']),
			list(spans['s']),
			flow_edges(flow))

		return MatrixRenderer._template.safe_substitute(
			source_nodes_values=json.dumps({
				'source_nodes': data['source_nodes']
			}),
			target_nodes_values=json.dumps({
				'target_nodes': data['target_nodes']
			}),
			links_values=json.dumps({
				'links': data['links']
			}),
			cell_size=20)
