from typing import Dict, Optional

import bokeh
import ipywidgets as widgets
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Spectral4
from bokeh.plotting import figure
from IPython.display import display
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only

output_notebook(bokeh.resources.INLINE)


class Logger(LightningLoggerBase):

  def __init__(self, metrics=[]):
    super().__init__()
    self._name = 'Logger'
    self._version = '0'

    # Initialise a plot.
    tooltips = [
        ('metric', '$name'),
        ('value', '@values'),
        ('epoch', '@steps'),
    ]
    tools = 'box_zoom, wheel_zoom, pan, hover, reset, save'
    self.plot = figure(
        plot_width=700, plot_height=400, tooltips=tooltips, tools=tools)

    # The names of metrics that are being logged.
    self.metrics = {}
    self.lines = {}
    self.sources = {}
    self.legend_items = []

    for metric, color in zip(metrics, Spectral4):
      self.metrics[metric] = {'values': [], 'steps': []}
      self.sources[metric] = ColumnDataSource(data=self.metrics[metric])
      self.lines[metric] = self.plot.line(
          x='steps',
          y='values',
          source=self.sources[metric],
          color=color,
          name=metric)
      self.legend_items.append((metric, [self.lines[metric]]))

    # Configure plot.
    self.plot.xaxis.axis_label = "Epoch"
    self.plot.yaxis.axis_label = "Metric"
    self.legend = Legend(items=self.legend_items, location="center")
    self.plot.add_layout(self.legend, 'above')
    self.plot.legend.orientation = "horizontal"

  @property
  def experiment(self) -> None:
    return None

  @rank_zero_only
  def log_hyperparams(self, params):
    # params is an argparse.Namespace
    # your code to record hyperparameters goes here{}
    pass

  @rank_zero_only
  def log_metrics(self,
                  metrics: Dict[str, float],
                  step: Optional[int] = None) -> None:
    for metric in metrics:
      self.push_metric(metric, metrics[metric], step)

  @rank_zero_only
  def save(self):
    # Optional. Any code necessary to save logger data goes here
    pass

  @rank_zero_only
  def finalize(self, status):
    # Optional. Any code that needs to be run after training
    # finishes goes here
    pass

  @property
  def name(self) -> str:
    return self._name

  @property
  def version(self) -> int:
    return self._version

  def push_metric(self, metric, value, step=None):
    if metric not in self.metrics:
      return
    elif len(self.metrics[metric]['values']) == 0:
      self.metrics[metric] = {
          'values': [value],
          'steps': [step if step is not None else 0]
      }
    else:
      self.metrics[metric]['values'].append(value)
      if step is not None:
        self.metrics[metric]['steps'].append(step)
      else:
        self.metrics[metric]['steps'].append(self.metrics[metric]['steps'][-1] +
                                             1)
    self.sources[metric].data = self.metrics[metric]

    push_notebook(handle=self.handle)

  def draw(self):
    # Create a Jupyter widget.
    self.widget = widgets.Output()
    display(self.widget)
    # Draw the plot in the widget.
    with self.widget:
      self.handle = show(self.plot, notebook_handle=True)