def flow_edges(flow, tolerance=0):
	if flow['type'] == 'injective':

		for t, (s, f) in enumerate(zip(flow['target'], flow['flow'])):
			if s >= 0 and f > tolerance:
				yield int(t), int(s), float(f)

	elif flow['type'] == 'sparse':

		for t, s, f in zip(flow['source'], flow['target'], flow['flow']):
			if f > tolerance:
				yield int(t), int(s), float(f)

	elif flow['type'] == 'dense':

		m = flow['flow']
		for t in range(m.shape[0]):
			for s in range(m.shape[1]):
				f = m[t, s]
				if f > tolerance:
					yield int(t), int(s), float(f)

	else:
		raise ValueError(flow['type'])
