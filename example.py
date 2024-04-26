from shiny import ui, render, App
from matplotlib import pyplot as plt

app_ui = ui.page_fluid(
    ui.output_plot("a_scatter_plot"),
)

def server(input, output, session):
    @output
    @render.plot
    def a_scatter_plot():
        return plt.scatter([1,2,3], [5, 2, 3])

app = App(app_ui, server)