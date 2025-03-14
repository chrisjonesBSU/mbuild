import ipywidgets as widgets
from IPython.display import clear_output, display
from traitlets import HasTraits


class Visualize(HasTraits):
    def __init__(self, compound):
        self.compound = compound
        self.slider = widgets.IntSlider(
            min=-1, max=compound.n_particles - 1, value=-1, description="Index:"
        )
        self.output = widgets.Output()  # Capture output

        interactive_widget = widgets.interactive(self.visualize, index=self.slider)

        display(interactive_widget)

    def visualize(self, index):
        clear_output(wait=True)

        if index >= 0:
            change_particle = [p for p in self.compound.particles()][index]
            old_name = change_particle.name
            change_particle.name = "X"
            self.compound.visualize().show()
            change_particle.name = old_name
        else:  # index == 0
            self.compound.visualize().show()
