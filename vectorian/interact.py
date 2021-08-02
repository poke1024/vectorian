import ipywidgets as widgets
import matplotlib.pyplot as plt
import time

import vectorian.alignment
import vectorian.embeddings
import vectorian.metrics
import vectorian.session
import vectorian.core as core


ROOT_LEVEL_STYLE = {'description_width': '10em'}


def make_root_label(s):
	return widgets.Label(s, layout=widgets.Layout(width='10em', display='flex', justify_content='flex-end'))


class FineTuneableWidget:
	_level = 0

	def __init__(self, iquery, fix_to=None, default=None, default_options=None):
		if default is None:
			default = self._default
		kwargs = dict(
			options=[x[0] for x in self._types],
			value=default if fix_to is None else fix_to,
			description=self._description,
			disabled=fix_to is not None,
			layout={'width': '25em'},
			style=ROOT_LEVEL_STYLE if self._level == 0 else {})

		layout = self.layout
		if layout is not None:
			kwargs['layout'] = layout

		self._type = widgets.Dropdown(**kwargs)

		self._iquery = iquery

		self._instantiate_fine_tune(self._type.value, default_options)

		self._type.observe(self.on_changed, names='value')

		box_type = getattr(self, '_box', widgets.VBox)
		self._box = box_type([self._type, self._fine_tune.widget])

	def _instantiate_fine_tune(self, name, options=None):
		if options is None:
			options = {}
		i = [x[0] for x in self._types].index(name)
		self._fine_tune = self._types[i][1](self._iquery, **options)

	def on_changed(self, change):
		self._instantiate_fine_tune(change.new)
		self._box.children = [self._type, self._fine_tune.widget]
		self._iquery.on_changed()

	def make(self):
		return self._fine_tune.make()

	@property
	def value(self):
		return self._type.value

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
			step=0.01,
			description='gamma:',
			disabled=False,
			layout={'width': '10em'})

		self._vbox = widgets.VBox([self._p, self._scale])

	@property
	def widget(self):
		return self._vbox

	def make(self):
		return vectorian.metrics.ModifiedVectorSimilarity(
			vectorian.metrics.PNormDistance(p=self._p.value),
			vectorian.metrics.RadialBasis(self._scale.value)
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

	def describe(self):
		return f"**{self.value.lower()} similarity**"

	@property
	def layout(self):
		return {'width': '15em'}


class EmbeddingMixerWidget:
	def __init__(self, iquery):
		self._iquery = iquery

		names = []
		for x in sorted(iquery.session.embeddings.keys()):
			names.append(x)

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
		for x, _ in iquery.ordered_embedding:
			options.append(x)

		self._embedding = widgets.Dropdown(
			options=options,
			value=options[default],
			description='',
			disabled=False,
			layout={'width': '15em'})

		self._embedding.observe(self.on_changed, names='value')

		self._vbox = widgets.VBox([
			self._embedding
		])

	def on_changed(self, changed):
		self._iquery.on_changed()

	@property
	def widget(self):
		return self._vbox

	def make(self):
		return self._iquery.session.embeddings[self._embedding.value].factory

	def describe(self):
		return f"**{self._embedding.value}**"


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

	def describe(self):
		return f"{self._metric.describe()} over {self._embedding.describe()}"

	@property
	def weight(self):
		return self._weight.value


class TokenSimilarityMetricWidget:
	def __init__(self, iquery, similarity=None):
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
			layout={'width': '30em'},
			style=ROOT_LEVEL_STYLE)

		self._vbox = widgets.VBox([
			widgets.HBox([
				self._operator,
				self._operands_vbox]),
			self._falloff
		])

		self._num_operands = 1

		if similarity is not None:
			emb_i = [x for x, _ in iquery.ordered_embedding].index(
				similarity['embedding'].name)
			self._update_operand_widgets(default_embedding=emb_i)
		else:
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

		self._iquery.on_changed()

	def _update_operand_widgets(self, default_embedding=None):
		option = self._option_info(self._operator.value)
		max_i = len(self._iquery.session.embeddings) - 1

		self._operands = []
		if default_embedding is None:
			for i in range(self._num_operands):
				self._operands.append(
					TokenSimilarityAtomWidget(
						self._iquery,
						embedding=min(max_i, i),
						add_weight=option['weights']))
		else:
			assert self._num_operands == 1
			self._operands.append(
				TokenSimilarityAtomWidget(
					self._iquery,
					embedding=default_embedding,
					add_weight=option['weights']))

		if option['multiple']:
			add_operand = widgets.Button(
				description='',
				icon='plus-square')
			add_operand.on_click(self.on_add_operand)
			extra = [add_operand]
		else:
			extra = []

		self._operands_vbox.children = [x.widget for x in self._operands] + extra

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

	def describe(self):
		text = [self._operator.value.lower(), ", by employing "]
		for i, x in enumerate(self._operands):
			text.append(x.describe())
			if i > 0:
				text.append(" and ")
		if self._falloff.value != 1:
			text.append(". A **falloff** of **%.2f** is applied." % self._falloff)
		else:
			text.append(".")
		return "".join(text)

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
	def __init__(self, iquery, description, construct, default=0, max=1.0, step=0.01):
		self._construct = construct
		self._iquery = iquery

		self._cost = widgets.FloatSlider(
			value=default,
			min=0,
			max=max,
			step=step,
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
		self._iquery.on_changed()

	@property
	def widget(self):
		return self._vbox


class ConstantGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery, k=0.1):
		super().__init__(iquery, 'Cost:', vectorian.alignment.ConstantGapCost, default=k)

	def describe(self):
		return "**constant gap cost** of **%.2f**" % self._cost.value


class LinearGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery, k=0.1):
		super().__init__(iquery, 'Cost:', vectorian.alignment.LinearGapCost, default=k)

	def describe(self):
		return "**linear gap cost** of **%.2f**" % self._cost.value


class SmoothGapCostWidget(SlidingGapCostWidget):
	def __init__(self, iquery, k=3):
		super().__init__(iquery, 'Cutoff:', vectorian.alignment.smooth_gap_cost, default=k, max=k * 7, step=1)

	def describe(self):
		return "**smooth gap cost** with a 50%% penalty at **%.2f**" % self._cost.value


class GapCostWidget(FineTuneableWidget):
	_level = 1

	_description = 'Gap Type:'

	_types = [
		('Constant', ConstantGapCostWidget),
		('Linear', LinearGapCostWidget),
		('Exponential', SmoothGapCostWidget)
	]

	_default = 'Linear'

	_box = widgets.HBox

	def describe(self):
		return self._fine_tune.describe()


class GapMaskWidget:
	def __init__(self, iquery):
		self._s = widgets.Checkbox(value=True, description="document")
		self._t = widgets.Checkbox(value=True, description="query")
		self._hbox = widgets.HBox([make_root_label("Gap Mask:"), self._s, self._t])

	@property
	def widget(self):
		return self._hbox

	def get(self):
		return "".join([
			"s" if self._s.value else "",
			"t" if self._t.value else ""
		])

	def describe(self):
		text = " and ".join([f"**{x}**" for x in self.get()])
		return f" (applied to {text})"


def derive_gap_cost_args(gap_cost):
	args_s = gap_cost['s'].to_tuple()
	args_t = gap_cost['t'].to_tuple()
	if args_s != args_t:
		raise RuntimeError(
			f'cannot derive non-unified gap costs for s and t: {args_s} != {args_t}')
	if args_s[0] == 'exponential' and args_s[1] == 2:
		return {
			'default': 'Exponential',
			'default_options': {
				'k': int(1 / args_s[2])
			}
		}
	elif args_s[0] == 'linear':
		return {
			'default': 'Linear',
			'default_options': {
				'k': args_s[1]
			}
		}
	elif args_s[0] == 'constant':
		return {
			'default': 'Constant',
			'default_options': {
				'k': args_s[1]
			}
		}
	else:
		raise RuntimeError(f'cannot decompose gap cost specification {args_s}')


class AlignmentAlgorithmWidget:
	def __init__(self, iquery, klass, alignment=None, indent='5em', similarity=None):
		if alignment is not None:
			gap_cost_widget_args = derive_gap_cost_args(alignment.gap)
		else:
			gap_cost_widget_args = {
				'default': 'Exponential'
			}

		self._gap_cost = GapCostWidget(iquery, **gap_cost_widget_args)
		self._gap_mask = GapMaskWidget(iquery)

		self._klass = klass
		self._token_sim = TokenSimilarityMetricWidget(iquery, similarity)

		parameters = [self._gap_cost.widget, self._gap_mask.widget]
		if parameters is None:
			self._vbox = widgets.VBox([
				self._token_sim.widget])
		else:
			for p in parameters:
				p.layout = widgets.Layout(margin=f'0 0 0 {indent}')
			self._vbox = widgets.VBox([
				self._token_sim.widget,
				make_root_label('Alignment Args:'),
				*parameters])

	@property
	def widget(self):
		return self._vbox

	def make_token_sim(self):
		return self._token_sim.make()

	def describe_token_sim(self):
		return self._token_sim.describe()

	def make(self):
		gap = self._gap_cost.make()
		mask = self._gap_mask.get()
		return self._klass(
			gap={
				's': gap if 's' in mask else vectorian.alignment.ConstantGapCost(0),
				't': gap if 't' in mask else vectorian.alignment.ConstantGapCost(0)
			})

	def describe_alignment(self):
		return "with " + self._gap_cost.describe() + self._gap_mask.describe()


class GlobalAlignmentWidget(AlignmentAlgorithmWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, klass=vectorian.alignment.GlobalAlignment, **kwargs)


class SemiGlobalAlignmentWidget(AlignmentAlgorithmWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, klass=vectorian.alignment.SemiGlobalAlignment)


class LocalAlignmentWidget(AlignmentAlgorithmWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, klass=vectorian.alignment.LocalAlignment)


class WordMoversDistanceWidget(AlignmentAlgorithmWidget):
	_variants = [
		'wmd/bow',
		'wmd/nbow',
		'rwmd/nbow',
		'rwmd/nbow/distributed',
		'rwmd/bow/fast'
	]

	def __init__(self, iquery, alignment=None, **kwargs):
		if alignment is not None:
			default_variant = alignment.builtin_name
		else:
			default_variant = "wmd/nbow"

		self._variant = widgets.Dropdown(
			options=self._variants,
			value=default_variant,
			description="Variant:",
			disabled=False)

		self._extra_mass_penalty = widgets.FloatText(
			value=-1,
			description='Extra Mass Penalty:',
			disabled=False,
			style={'description_width': 'initial'})

		super().__init__(iquery, [
			self._variant,
			self._extra_mass_penalty
		], indent='10em', **kwargs)

	def make(self):
		variant = self._variant.value.split("/", 1)
		if variant[0] == 'wmd':
			return vectorian.alignment.WordMoversDistance.wmd(
				variant[1], extra_mass_penalty=self._extra_mass_penalty.value)
		elif variant[0] == 'rwmd':
			return vectorian.alignment.WordMoversDistance.rwmd(
				variant[1], extra_mass_penalty=self._extra_mass_penalty.value)
		else:
			raise ValueError(self._variant.value)

	def describe_alignment(self):
		return f"in the **{self._variant.value}** variant"


class WordRotatorsDistanceWidget(AlignmentAlgorithmWidget):
	def __init__(self, iquery, alignment=None, **kwargs):
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

		super().__init__(iquery, [
			self._normalize_magnitudes,
			self._extra_mass_penalty
		], indent='10em', **kwargs)

	def make(self):
		return vectorian.alignment.WordRotatorsDistance(
			normalize_magnitudes=self._normalize_magnitudes.value,
			extra_mass_penalty=self._extra_mass_penalty.value)

	def describe_alignment(self):
		if self._normalize_magnitudes.value:
			return "with normalized magnitudes"
		else:
			return "without normalized magnitudes"


class AlignmentWidget(FineTuneableWidget):
	_description = 'Alignment:'

	_types = [
		('Global Alignment', GlobalAlignmentWidget),
		('Semiglobal Alignment', SemiGlobalAlignmentWidget),
		('Local Alignment', LocalAlignmentWidget),
		('Word Movers Distance (WMD)', WordMoversDistanceWidget),
		('Word Rotators Distance (WRD)', WordRotatorsDistanceWidget)
	]

	_default = 'Local Alignment'

	def __init__(self, iquery, alignment=None, similarity=None, **kwargs):
		if 'default_options' not in kwargs:
			kwargs['default_options'] = {}

		if alignment is not None:
			def map_alignment(args):
				locality = args['options']['locality']
				to_str = dict((v, k) for k, v in core.pyalign.Locality.__members__.items())
				return f'{to_str[locality].capitalize()} Alignment'

			mapping = {
				'pyalign': map_alignment,
				'word-movers-distance': lambda args: 'Word Movers Distance (WMD)',
				'word-rotators-distance': lambda args: 'Word Rotators Distance (WRD)'
			}

			args = alignment.to_args(iquery.partition)
			kwargs['default'] = mapping[args['algorithm']](args)
			kwargs['default_options']['alignment'] = alignment

		if similarity is not None:
			kwargs['default_options']['similarity'] = similarity

		super().__init__(iquery, **kwargs)

	def make_alignment(self):
		return self._fine_tune.make()

	def make_token_sim(self):
		return self._fine_tune.make_token_sim()

	def make(self):
		return vectorian.metrics.NetworkFlowSimilarity(
			token_sim=self.make_token_sim(),
			flow_strategy=self.make_alignment())

	def describe(self):
		return ''.join([
			f"**alignment** using **{self.value}** over token similarities. ",
			f"{self.value} is employed {self._fine_tune.describe_alignment()}. ",
			"Token similarity is computed through ", self._fine_tune.describe_token_sim()])


class TagWeightedAlignmentWidget:
	def __init__(self, iquery, tag_weights=None, **kwargs):
		self._pos_mismatch_penalty = widgets.FloatSlider(
			value=1,
			min=0,
			max=1,
			step=0.1,
			description='POS Mismatch Penalty:',
			disabled=False,
			style=ROOT_LEVEL_STYLE)

		if tag_weights is None:
			# weights from Batanovic et al.
			self._tag_weights = dict((k, float(v)) for k, v in [
				('CC', '0.7'), ('CD', '0.8'), ('DT', '0.7'), ('EX', '0.7'), ('FW', '0.7'), ('IN', '0.7'), ('JJ', '0.7'),
				('JJR', '0.7'), ('JJS', '0.8'), ('LS', '0.7'), ('MD', '1.2'), ('NN', '0.8'), ('NNS', '1.0'), ('NNP', '0.8'),
				('NNPS', '0.8'), ('PDT', '0.7'), ('POS', '0.7'), ('PRP', '0.7'), ('PRP$', '0.7'), ('RB', '1.3'), ('RBR', '1.2'),
				('RBS', '1.0'), ('RP', '1.2'), ('SYM', '0.7'), ('TO', '0.8'), ('UH', '0.7'), ('VB', '1.2'), ('VBD', '1.2'),
				('VBG', '1.1'), ('VBN', '0.8'), ('VBP', '1.2'), ('VBZ', '1.2'), ('WDT', '0.7'), ('WP', '0.7'), ('WP$', '0.7'),
				('WRB', '1.3')])
		else:
			self._tag_weights = tag_weights

		'''
		self._tag_weights = widgets.Dropdown(
			options=['Off', 'POST STSS'],
			value='POST STSS',
			description='Tag Weights:',
			disabled=False,
			style=ROOT_LEVEL_STYLE)
			'''

		self._similarity_threshold = widgets.FloatSlider(
			value=0.2,
			min=0,
			max=1,
			step=0.1,
			description='Similarity Threshold:',
			disabled=False,
			style=ROOT_LEVEL_STYLE)

		self._alignment = AlignmentWidget(iquery, **kwargs)

		self._vbox = widgets.VBox([
			self._pos_mismatch_penalty,
			#self._tag_weights,
			self._similarity_threshold,
			self._alignment.widget
		])

	def make(self):
		return vectorian.metrics.NetworkFlowSimilarity(
			token_sim=self._alignment.make_token_sim(),
			flow_strategy=self._alignment.make_alignment(),
			tag_weights=self._tag_weights)

	@property
	def widget(self):
		return self._vbox

	def describe(self):
		assignments = []
		for k, v in self._tag_weights.items():
			assignments.append(f"{k}={v}")
		if len(assignments) <= 3:
			desc_text = ", ".join(assignments)
		else:
			desc_text = ", ".join(assignments) + ", ..."

		return f"**tag-weighted** ({desc_text}) " + self._alignment.describe()


class PartitionEmbeddingWidget:
	def __init__(self, iquery):
		self._iquery = iquery
		self._encoders = self._iquery.partition_encoders
		keys = sorted(self._encoders.keys())
		self._widget = widgets.Dropdown(
			options=keys,
			value=keys[0],
			description='Model:',
			disabled=False,
			layout={'width': '25em'})

	def make(self):
		return vectorian.metrics.SpanEmbeddingSimilarity(
			self._encoders[self._widget.value].to_cached())

	def describe(self):
		return f"partition embeddings using {self._widget.value}."

	@property
	def widget(self):
		return self._widget


class PartitionMetricWidget(FineTuneableWidget):
	_description = 'Strategy:'

	_types = [
		('Alignment', AlignmentWidget),
		('Tag-Weighted Alignment', TagWeightedAlignmentWidget),
		('Partition Embedding', PartitionEmbeddingWidget)
	]

	_default = 'Alignment'

	def describe(self):
		return "Partition similarity is computed via " + self._fine_tune.describe()


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
			layout=widgets.Layout(margin='1em 0 0 0'))

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
			style=ROOT_LEVEL_STYLE,
			continuous_update=False)
		self._query.observe(self.on_search, 'value')

		self._submit_query = widgets.Button(
			description='Search',
			button_style='success',
			icon='search')
		self._submit_query.on_click(self.on_search)

		self._partition = PartitionWidget(iquery)
		self._sentence = PartitionMetricWidget(iquery)

		self._progress = widgets.FloatProgress(
			value=0, min=0, max=1, description='',
			layout=widgets.Layout(width='100%', visibility='hidden'))

		self._num_results_slider = widgets.IntSlider(
			description='Results:', value=1, min=1, max=100, step=1,
			layout={'width': '40em'}, style=ROOT_LEVEL_STYLE)

		self._location_formatter = vectorian.render.location.LocationFormatter()

		self._render = MatchRenderWidget(iquery)
		self._results = widgets.HTML(value='')

		self._results_format = 'excerpt'
		self._results_obj = None

		widgets.Accordion(children=[], titles=[])

		self._vbox = widgets.VBox([
			widgets.HBox([self._query, self._submit_query]),
			self._num_results_slider,
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

	def on_changed(self):
		self._results.value = ''

	def on_search(self, change):
		self.search()

	def search(self):
		self.on_changed()

		debug_path = None

		if debug_path:
			def debug(hook, data):
				if hook == 'alignment/word-rotators-distance/solver':
					import numpy as np
					with open(debug_path, "a") as f:
						for k, v in data.items():
							if isinstance(v, np.ndarray):
								f.write(f"{k}: {v}\n")
							else:
								f.write(f"{k}: {v}\n")
						f.write("-" * 80)
						f.write("\n")
		else:
			debug = None

		r = self.index.find(
			self._query.value, n=self._num_results_slider.value,
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

	@property
	def partition_encoders(self):
		return {}

	def set_index(self, index):
		pass

	def on_changed(self):
		self._widget.on_changed()

	def set_format(self, fmt):
		self._widget.set_format(fmt)

	@property
	def widget(self):
		return self._widget.widget

	@property
	def ordered_embedding(self):
		return sorted(list(self._session.embeddings.items()), key=lambda x: x[0])
