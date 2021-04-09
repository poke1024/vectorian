import string
import base64


class VegaRenderer:
	script_code = string.Template('''
	(function(spec) {
		var view = new vega.View(vega.parse(spec), {
			renderer: '${renderer}',
			container: '#${div_id}',
			hover: true
		});
		view.runAsync().then(function() {
			parent.document.getElementById('${iframe_id}').onload();
		});
	}) (JSON.parse(atob('${vega_spec}')));
	''')

	def __init__(self):
		self._data = {}

	def _make_vega_spec(self, match):
		raise NotImplementedError()

	def write_head(self, doc):
		doc, tag, text = doc.tagtext()
		with tag('script', src='https://vega.github.io/vega/vega.min.js'):
			pass

	def write_match(self, doc, match, fetch_id):
		doc, tag, text = doc.tagtext()
		div_id = fetch_id()
		with tag('div', id=div_id):
			self._data[div_id] = match

	def write_script(self, doc, iframe_id):
		doc, tag, text = doc.tagtext()
		with tag('script'):
			for div_id, match in self._data.items():
				text(VegaRenderer.script_code.safe_substitute(
					renderer='canvas',  # canvas or svg
					div_id=div_id,
					iframe_id=iframe_id,
					vega_spec=base64.b64encode(
						self._make_vega_spec(match).encode('utf8')).decode('utf8')
				))

