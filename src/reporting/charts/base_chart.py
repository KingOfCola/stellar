import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from reportlab.graphics.shapes import Drawing
from reportlab.platypus import Flowable


class BaseChart:
    def __init__(self, **kwargs):
        self.fig, self.axes = plt.subplots(**kwargs)

    def plot(self):
        """
        Plot the chart using matplotlib.
        """
        # This method should be overridden by subclasses to implement specific plotting logic
        raise NotImplementedError("Subclasses should implement this method.")

    def to_rlg(self):
        """
        Convert the matplotlib figure to a ReportLab drawing object.
        """
        # Save the figure to a BytesIO object in SVG format
        imgdata = BytesIO()
        self.fig.savefig(imgdata, format="svg")
        imgdata.seek(0)

        # Convert the SVG data to a ReportLab drawing object
        drawing = svg2rlg(imgdata)
        return drawing


class SvgFlowable(Flowable):
    """Convert byte stream containing SVG into a Reportlab Flowable."""

    def __init__(self, rlg: BytesIO) -> None:
        """Convert SVG to RML drawing on initializtion."""
        self.drawing: Drawing = rlg
        self.width: int = self.drawing.minWidth()
        self.height: int = self.drawing.height
        self.drawing.setProperties({"vAlign": "CENTER", "hAlign": "CENTER"})

    def wrap(self, *_args):
        """Return diagram size."""
        return (self.width, self.height)

    def draw(self) -> None:
        """Render the chart."""
        renderPDF.draw(self.drawing, self.canv, 0, 0)
