# © All rights reserved. University of Edinburgh, United Kingdom
# IDEAL Project, 2018

rc = {'axes.labelweight': 'bold',
      'axes.titleweight': 'bold',
      'figure.dpi': 120,
      'figure.titleweight': 'bold',
      'axes.linewidth': .5,
      'figure.figsize': (8,3.5),
      'savefig.format': 'eps',
      'savefig.dpi': 320}

sns.set(context='paper', style="ticks", rc=rc)
sns.set_context('paper', font_scale=1.1, rc=rc)
sns.set_palette("colorblind")

single_figure = (6,3.5)
double_figure = (8,3.5)
