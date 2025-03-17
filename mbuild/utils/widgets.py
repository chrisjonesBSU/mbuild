import gc

import IPython.display
import ipywidgets as widgets
from IPython.display import clear_output, display
from traitlets import HasTraits


class Visualize(HasTraits):
    def __init__(self, compound):
        self.compound = compound
        self.slider = widgets.IntSlider(
            min=-1, max=compound.n_particles - 1, value=-1, description="P. Index:"
        )
        self.output = widgets.Output()
        display(self.slider)
        display(self.output)
        self.slider.observe(self.on_slider_change, names="value")
        self.visualize(index=-1)

    def on_slider_change(self, change):
        self.visualize(change.new)

    def visualize(self, index):
        # Force a more aggressive cleanup
        IPython.display.clear_output(wait=True)
        gc.collect()

        with self.output:
            clear_output(wait=True)
            # Create visualization with memory management
            try:
                if index >= 0:
                    change_particle = [p for p in self.compound.particles()][index]
                    old_name = change_particle.name
                    change_particle.name = "X"
                    viz = self.compound.visualize()
                    viz.show()
                    change_particle.name = old_name
                    del viz
                    # del temp_comp
                else:  # index < 0
                    viz = self.compound.visualize()
                    viz.show()
                    del viz
            finally:
                gc.collect()
