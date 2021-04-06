import ipywidgets as widgets
import matplotlib.pyplot as plt
import time

import vectorian.alignment
import vectorian.metrics
import vectorian.session


ROOT_LEVEL_STYLE = {'description_width': '10em'}


def make_root_label(s):
	return widgets.Label(s, layout=widgets.Layout(width='10em', display='flex', justify_content='flex-end'))


class FineTuneableWidget:
	_level = 0

	def __init__(self, iquery, fix_to=None):
		kwargs = dict(
			options=[x[0] for x in self._types],
			value=self._default if fix_to is None else fix_to,
			description=self._description,
			disabled=fix_to is not None,
			layout={'width': '25em'},
			style=ROOT_LEVEL_STYLE if self._level == 0 else {})

		layout = self.layout
		if layout is not None:
			kwargs['layout'] = layout

		self._type = widgets.Dropdown(**kwargs)

		self._iquery = iquery

		self._instantiate_fine_tune(self._type.value)

		self._type.observe(self.on_changed, names='value')

		box_type = getattr(self, '_box', widgets.VBox)
		self._box = box_type([self._type, self._fine_tune.widget])

	def _instantiate_fine_tune(self, name):
		i = [x[0] for x in self._types].index(name)
		self._fine_tune = self._types[i][1](self._iquery)

	def on_changed(self, change):
		self._iquery.clear_output()
		self._instantiate_fine_tune(change.new)
		self._box.children = [self._type, self._fine_tune.widget]

	def make(self):
		return self._fine_tune.make()

	@property
	def widget(self):
		return self._box

	@property
	def layout(self):
		return None


class CosineMetricWidget:
	def __init__(self, iquery):
		self._vbox = widgets.VBox([])

	@property
	def widget(self):
		return self._vbox

	def make(self):
		return vectorian.metrics.CosineSimilarity()


class ImprovedSqrtCosineMetricWidget:
	def __init__(self, iquery):
		self._vbox = widgets.VBox([])

	@property
	def widget(self):
		return self._vbox

	def make(self):
		return vectorian.metrics.ImprovedSqrtCosineSimilarity()


class PNormWidget:
	def __init__(self, iquery):
		self._p = widgets.BoundedFloatText(
			value=2,
			min=1e-4,
			max=10,
			step=0.25,
			description='p:',
			disabled=False,
			layout={'width': '10em'})

		self._scale = widgets.BoundedFloatText(
			value=1,
			min=1e-4,
			max=1000,
			step=0.25,
			description='Scale:',
			disabled=False,
			layout={'width': '10em'})

		self._vbox = widgets.VBox([self._p, self._scale])

	@property
	def widget(self):
		return self._vbox

	def make(self):
		return vectorian.metrics.MetricModifier(
			vectorian.metrics.PNormDistance(p=self._p.value),
			[
				vectorian.metrics.Scale(self._scale.value),
				vectorian.metrics.DistanceToSimilarity()
			]
		)


class VectorMetricWidget(FineTuneableWidget):
	_description = ''

	_types = [
		('Cosine', CosineMetricWidget),
		('P-Norm', PNormWidget),
		('Improved Sqrt Cosine', ImprovedSqrtCosineMetricWidget)
	]

	_default = 'Cosine'

	def make(self):
		return self._fine_tune.make()

	@property
	def layout(self):
		return {'width': '15em'}


class EmbeddingMixerWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		names = []
		for x in iquery.session.embeddings:
			names.append(x.name)

		items = []
		for name in names:
			items.append(widgets.Checkbox(
				value=False,
				description=name,
				disabled=False,
				indent=False
			))
		self._items = items

		# self._grid = widgets.GridBox(items, layout=widgets.Layout(
		#    grid_template_columns="repeat(2, 40em)")))

		self._vbox = widgets.VBox(items)

	@property
	def widget(self):
		return self._vbox


class EmbeddingWidget:
	def __init__(self, iquery, default=0):
		self._iquery = iquery

		options = []
		for x in iquery.session.embeddings:
			options.append(x.name)

		self._embedding = widgets.Dropdown(
			options=options,
			value=options[default],
			description='',
			disabled=False,
			layout={'width': '15em'})

		# self._mixer = EmbeddingMixerWidget(iquery)

		self._vbox = widgets.VBox([
			self._embedding
		])

	@property
	def widget(self):
		return self._vbox

	def make(self):
		for x in self._iquery.session.embeddings:
			if x.name == self._embedding.value:
				return x


class TokenSimilarityAtomWidget:
	def __init__(self, iquery, embedding=0, add_weight=False):
		self._metric = VectorMetricWidget(iquery)

		self._embedding = EmbeddingWidget(
			iquery, default=embedding)

		items = [
			self._metric.widget,
			widgets.Label('on'),
			self._embedding.widget]

		if add_weight:
			self._weight = widgets.FloatSlider(
				value=1,
				min=0,
				max=1,
				step=0.05,
				description='',
				disabled=False,
				layout={'width': '15em'})
			items.append(self._weight)

		self._hbox = widgets.HBox(items)

	@property
	def widget(self):
		return self._hbox

	def make(self):
		return vectorian.metrics.TokenSimilarity(
			self._embedding.make(),
			self._metric.make())

	@property
	def weight(self):
		return self._weight.value


class TokenSimilarityMetricWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		self._options = [
			{
				'name': 'One Embedding',
				'multiple': False,
				'weights': False,
				'make': self._make_one
			},
			{
				'name': 'Mixed Embeddings',
				'multiple': True,
				'weights': True,
				'make': self._make_mixed
			},
			{
				'name': 'Maximum Similarity',
				'multiple': True,
				'weights': False,
				'make': self._make_max
			},
			{
				'name': 'Minimum Similarity',
				'multiple': True,
				'weights': False,
				'make': self._make_min
			}
		]

		self._operator = widgets.Dropdown(
			options=[x['name'] for x in self._options],
			value=self._options[0]['name'],
			description='Similarity:',
			disabled=False,
			layout={'width': '20em'},
			style=ROOT_LEVEL_STYLE)

		self._operands_vbox = widgets.VBox([])

		self._falloff = widgets.FloatLogSlider(
			value=1,
			base=2,
			min=-3,
			max=2,
			step=0.1,
			description='Falloff:',
			disabled=False,
			layout={'width': '20em'},
			style=ROOT_LEVEL_STYLE)

		self._vbox = widgets.VBox([
			widgets.HBox([
				self._operator,
				self._operands_vbox]),
			self._falloff
		])

		self._num_operands = 1
		self._update_operand_widgets()

		self._operator.observe(self.on_changed, names='value')

	def _option_info(self, name):
		k = [x['name'] == name for x in self._options].index(True)
		return self._options[k]

	def on_changed(self, changed):
		option = self._option_info(changed.new)

		if option['multiple']:
			n = len(self._iquery.session.embeddings)
		else:
			n = 1

		self._num_operands = n

		self._update_operand_widgets()

	def _update_operand_widgets(self):
		option = self._option_info(self._operator.value)
		max_i = len(self._iquery.session.embeddings) - 1

		self._operands = []
		for i in range(self._num_operands):
			self._operands.append(
				TokenSimilarityAtomWidget(
					self._iquery,
					embedding=min(max_i, i),
					add_weight=option['weights']))

		if option['multiple']:
			add_operand = widgets.Button(
				description='',
				icon='plus-square')
			add_operand.on_click(self.on_add_operand)
			extra = [add_operand]
		else:
			extra = []

		self._operands_vbox.children = [
										   x.widget for x in self._operands] + extra

	def on_add_operand(self, changed):
		self._num_operands += 1
		self._update_operand_widgets()

	@property
	def widget(self):
		return self._vbox

	def make(self):
		option = self._option_info(self._operator.value)
		operands = [x.make() for x in self._operands]
		sim = option['make'](operands)
		if self._falloff.value != 1:
			sim = vectorian.metrics.UnaryTokenSimilarityModifier(
				sim, [vectorian.metrics.Power(self._falloff.value)])
		return sim

	def _make_one(self, operands):
		return operands[0]

	def _make_mixed(self, operands):
		return vectorian.metrics.MixedTokenSimilarity(
			operands, weights=[x.weight for x in self._operands])

	def _make_max(self, operands):
		return vectorian.metrics.MaximumTokenSimilarity(operands)

	def _make_min(self, operands):
		return vectorian.metrics.MinimumTokenSimilarity(operands)


class SlidingGapCostWidget:
	def __init__(self, iquery, description, construct, max=1.0):
		self._construct = construct

		self._cost = widgets.FloatSlider(
			value=0,
			min=0,
			max=max,
			step=0.01,
			description=description,
			disabled=False)

		'''
		self._cost = widgets.BoundedFloatText(
			value=0,
			min=0,
			max=max,
			step=0.1,
			description=description,
			disabled=False)
		'''

		self._plot = widgets.Image(
			value=b'',
			format='png',
			width=300,
			height=400,
		)

		self.update_plot()
		self._cost.observe(self.on_changed, names='value')

		self._vbox = widgets.VBox([
			self._cost, self._plot], layout=widgets.Layout(border='solid'))

	def make(self):
		return self._construct(self._cost.value)

	def update_plot(self):
		fig, ax = plt.subplots(1, 1, figsize=(5, 2))
		im_data = self.make().plot_to_image(
			fig, ax, 20, format='png')
		plt.close()
		self._plot.value = im_data

	def on_changed(self, change):
		# cost = change.new
		self.update_plot()
		self._iquery.clear_output()

	@property
	def widget(self):
		return self._vbox


class ConstantGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery):
		super().__init__(iquery, 'Cost:', vectorian.alignment.ConstantGapCost)


class LinearGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery):
		super().__init__(iquery, 'Cost:', vectorian.alignment.LinearGapCost)


class ExponentialGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery):
		super().__init__(iquery, 'Cutoff:', vectorian.alignment.ExponentialGapCost, max=20)


class GapCostWidget(FineTuneableWidget):
	_level = 1

	_description = 'Gap Type:'

	_types = [
		('Constant', ConstantGapCostWidget),
		('Linear', LinearGapCostWidget),
		('Exponential', ExponentialGapCostWidget)
	]

	_default = 'Linear'

	_box = widgets.HBox


class AlignmentAlgorithmWidget:
	def __init__(self, iquery, parameters, indent='5em'):
		self._token_metric = TokenSimilarityMetricWidget(iquery)

		if parameters is None:
			self._vbox = widgets.VBox([
				self._token_metric.widget])
		else:
			parameters.layout = widgets.Layout(margin=f'0 0 0 {indent}')
			self._vbox = widgets.VBox([
				self._token_metric.widget,
				make_root_label('Alignment Args:'),
				parameters])

	@property
	def widget(self):
		return self._vbox

	def make_token_metric(self):
		return self._token_metric.make()


class NeedlemanWunschWidget(AlignmentAlgorithmWidget):
	def __init__(self, iquery):
		self._gap_cost = GapCostWidget(iquery, fix_to="Linear")
		super().__init__(iquery, self._gap_cost.widget)

	def make(self):
		return vectorian.alignment.NeedlemanWunsch(
			gap=self._gap_cost.make().to_scalar())


class SmithWatermanWidget(AlignmentAlgorithmWidget):
	def __init__(self, iquery):
		self._gap_cost = GapCostWidget(iquery, fix_to="Linear")
		self._zero = widgets.BoundedFloatText(
			value=0.25,
			min=0,
			max=1,
			step=0.1,
			description='Zero:',
			disabled=False)
		super().__init__(
			iquery,
			widgets.VBox([self._gap_cost.widget, self._zero]))

	def make(self):
		return vectorian.alignment.SmithWaterman(
			gap=self._gap_cost.make().to_scalar(), zero=self._zero.value)


class WatermanSmithBeyerWidget(AlignmentAlgorithmWidget):
	def __init__(self, iquery):
		self._gap_cost = GapCostWidget(iquery)
		self._zero = widgets.BoundedFloatText(
			value=0.25,
			min=0,
			max=1,
			step=0.1,
			description='Zero:',
			disabled=False)
		super().__init__(
			iquery,
			widgets.VBox(
				[self._gap_cost.widget, self._zero]))

	def make(self):
		return vectorian.alignment.WatermanSmithBeyer(
			gap=self._gap_cost.make(), zero=self._zero.value)


class WordMoversDistanceWidget(AlignmentAlgorithmWidget):
	_variants = [
		'wmd/kusner',
		'wmd/vectorian',
		'rwmd/kusner',
		'rwmd/jablonsky',
		'rwmd/vectorian'
	]

	def __init__(self, iquery):
		self._variant = widgets.Dropdown(
			options=self._variants,
			value="wmd/kusner",
			description="Variant:",
			disabled=False)

		self._extra_mass_penalty = widgets.FloatText(
			value=-1,
			description='Extra Mass Penalty:',
			disabled=False,
			style={'description_width': 'initial'})

		super().__init__(iquery, widgets.VBox([
			self._variant,
			self._extra_mass_penalty
		]), indent='10em')

	def make(self):
		variant = self._variant.value.split("/")
		if variant[0] == 'wmd':
			return vectorian.alignment.WordMoversDistance.wmd(
				variant[1], extra_mass_penalty=self._extra_mass_penalty.value)
		elif variant[0] == 'rwmd':
			return vectorian.alignment.WordMoversDistance.rwmd(
				variant[1], extra_mass_penalty=self._extra_mass_penalty.value)
		else:
			raise ValueError(self._variant.value)


class WordRotatorsDistanceWidget(AlignmentAlgorithmWidget):
	def __init__(self, iquery):
		self._normalize_magnitudes = widgets.Checkbox(
			value=False,
			description='Normalize Magnitudes',
			disabled=False,
			indent=False)

		self._extra_mass_penalty = widgets.FloatText(
			value=-1,
			description='Extra Mass Penalty:',
			disabled=False,
			style={'description_width': 'initial'})

		super().__init__(iquery, widgets.VBox([
			self._normalize_magnitudes,
			self._extra_mass_penalty
		]), indent='10em')

	def make(self):
		return vectorian.alignment.WordRotatorsDistance(
			normalize_magnitudes=self._normalize_magnitudes.value,
			extra_mass_penalty=self._extra_mass_penalty.value)


class AlignmentWidget(FineTuneableWidget):
	_description = 'Alignment:'

	_types = [
		('Needleman-Wunsch', NeedlemanWunschWidget),
		('Smith-Waterman', SmithWatermanWidget),
		('Waterman-Smith-Beyer', WatermanSmithBeyerWidget),
		('Word Movers Distance', WordMoversDistanceWidget),
		('Word Rotators Distance', WordRotatorsDistanceWidget)
	]

	_default = 'Waterman-Smith-Beyer'

	def make_alignment(self):
		return self._fine_tune.make()

	def make_token_metric(self):
		return self._fine_tune.make_token_metric()

	def make(self):
		return vectorian.metrics.AlignmentSentenceSimilarity(
			token_metric=self.make_token_metric(),
			alignment=self.make_alignment())


class TagWeightedAlignmentWidget():
	def __init__(self, iquery):
		self._pos_mismatch_penalty = widgets.FloatSlider(
			value=1,
			min=0,
			max=1,
			step=0.1,
			description='POS Mismatch Penalty:',
			disabled=False)

		self._tag_weights = widgets.Dropdown(
			options=['Off', 'POST STSS'],
			value='POST STSS',
			description='Tag Weights:',
			disabled=False)

		self._similarity_threshold = widgets.FloatSlider(
			value=0.2,
			min=0,
			max=1,
			step=0.1,
			description='Similarity Threshold:',
			disabled=False)

		self._alignment = AlignmentWidget(iquery)

		self._vbox = widgets.VBox([
			self._pos_mismatch_penalty,
			self._tag_weights,
			self._similarity_threshold,
			self._alignment.widget
		])

	def make(self):
		return vectorian.metrics.TagWeightedSentenceSimilarity(
			token_metric=self._alignment.make_token_metric(),
			alignment=self._alignment.make_alignment())

	@property
	def widget(self):
		return self._vbox


class SentenceEmbeddingWidget:
	# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
	_variants = [
		'stsb-roberta-large',
		'stsb-roberta-base',
		'stsb-bert-large',
		'stsb-distilbert-base'
	]

	def __init__(self, iquery):
		self._widget = widgets.Dropdown(
			options=self._variants,
			value='stsb-distilbert-base',
			description='Model:',
			disabled=False)

	def make(self):
		return vectorian.metrics.SentenceEmbeddingSimilarity()

	@property
	def widget(self):
		return self._widget


class SentenceMetricWidget(FineTuneableWidget):
	_description = 'Sentence Metric:'

	_types = [
		('Alignment', AlignmentWidget),
		('Tag-Weighted Alignment', TagWeightedAlignmentWidget),
		('Sentence Embedding', SentenceEmbeddingWidget)
	]

	_default = 'Alignment'


class PartitionWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		self._level = widgets.Dropdown(
			options=['sentence', 'token'],
			value='sentence',
			description='Partition:',
			disabled=False,
			style=ROOT_LEVEL_STYLE)

		self._window_size = widgets.BoundedIntText(
			value=1,
			min=1,
			max=1000,
			step=1,
			description='Window Size:',
			disabled=False,
			layout={'width': '10em'},
			style={'description_width': 'initial'})

		self._window_step = widgets.BoundedIntText(
			value=1,
			min=1,
			max=1000,
			step=1,
			description='Window Step:',
			disabled=False,
			layout={'width': '10em'},
			style={'description_width': 'initial'})

		self._hbox = widgets.HBox([
			self._level,
			self._window_size,
			self._window_step
		])

	def make(self):
		return vectorian.session.Partition(
			self._iquery.session, self._level.value,
			self._window_size.value, self._window_step.value)

	@property
	def widget(self):
		return self._hbox


class MatchRenderWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		flags = [
			'excerpt',
			'+annotations',
			'matrix'
		]

		items = []
		for f in flags:
			checkbox = widgets.ToggleButton(
				value=f == 'excerpt',
				description=f,
				disabled=False)
			checkbox.observe(self.on_changed, names='value')
			items.append(checkbox)
		self._items = items

		self._hbox = widgets.HBox(
			[make_root_label('Visualize:')] + items,
			style={'padding_top': '1em'})

	def on_changed(self, change):
		cmds = set()
		for item in self._items:
			if item.value:
				cmds.add(item.description)

		if '+annotations' in cmds:
			cmds.remove('+annotations')
			if 'excerpt' in cmds:
				cmds.remove('excerpt')
			cmds.add('excerpt +tags +metric +penalties')

		render_format = ', '.join(sorted(cmds))
		self._iquery.set_format(render_format)

	@property
	def widget(self):
		return self._hbox


class QueryWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		self._query = widgets.Text(
			value='',
			placeholder='Your Query',
			description='Query:',
			disabled=False,
			layout={'width': '40em'},
			style=ROOT_LEVEL_STYLE)
		self._query.on_submit(self.on_search)

		self._submit_query = widgets.Button(
			description='Search',
			button_style='success',
			icon='search')
		self._submit_query.on_click(self.on_search)

		self._partition = PartitionWidget(iquery)
		self._sentence = SentenceMetricWidget(iquery)

		self._progress = widgets.FloatProgress(
			value=0, min=0, max=1, description='',
			layout=widgets.Layout(width='100%', visibility='hidden'))

		self._location_formatter = vectorian.render.location.LocationFormatter()

		self._render = MatchRenderWidget(iquery)
		self._results = widgets.HTML(value='')

		self._results_format = 'excerpt'
		self._results_obj = None

		widgets.Accordion(children=[], titles=[])

		self._vbox = widgets.VBox([
			widgets.HBox([self._query, self._submit_query]),
			self._partition.widget,
			self._sentence.widget,
			self._render.widget,
			self._progress,
			self._results])

		self._task_start_time = None

	@property
	def partition(self):
		return self._partition.make()

	@property
	def sentence_metric(self):
		return self._sentence.make()

	@property
	def index(self):
		partition = self._iquery.session.partition(**self.partition.to_args())
		return partition.index(self.sentence_metric, self._iquery.nlp)

	def _update_progress(self, t):
		if time.time() - self._task_start_time > 1:
			self._progress.layout.visibility = 'visible'
			self._progress.value = self._progress.max * t

	def _run_task(self, task):
		self._task_start_time = time.time()
		try:
			result = task(self._update_progress)
		finally:
			self._progress.layout.visibility = 'hidden'
		return result

	def _make_result(self, *args, **kwargs):
		return vectorian.session.LabResult(
			*args, **kwargs,
			renderers=[vectorian.render.excerpt.ExcerptRenderer()],
			location_formatter=self._location_formatter)

	def clear_output(self):
		self._results.value = ''

	def on_search(self, change):
		self.search()

	def search(self):
		self.clear_output()

		debug = None

		def debug(hook, data):
			if hook == 'alignment/word-rotators-distance/solver':
				import numpy as np
				with open("/Users/arbeit/Desktop/debug.txt", "a") as f:
					for k, v in data.items():
						if isinstance(v, np.ndarray):
							f.write(f"{k}: {v}\n")
						else:
							f.write(f"{k}: {v}\n")
					f.write("-" * 80)
					f.write("\n")

		r = self.index.find(
			self._query.value, n=1,
			run_task=self._run_task,
			make_result=self._make_result,
			debug=None)

		self._results_obj = r
		if self._results_obj:
			self._results.value = self._results_obj.format(self._results_format)._repr_html_()

	def set_format(self, fmt):
		self._results_format = fmt
		if self._results_obj:
			self._results.value = self._results_obj.format(self._results_format)._repr_html_()

	@property
	def widget(self):
		return self._vbox


class InteractiveQuery:
	def __init__(self, session, nlp):
		self._session = session
		self._nlp = nlp
		self._widget = QueryWidget(self)

	@property
	def session(self):
		return self._session

	@property
	def nlp(self):
		return self._nlp

	def set_index(self, index):
		pass

	def clear_output(self):
		self._widget.clear_output()

	def set_format(self, fmt):
		self._widget.set_format(fmt)

	@property
	def widget(self):
		return self._widget.widget
