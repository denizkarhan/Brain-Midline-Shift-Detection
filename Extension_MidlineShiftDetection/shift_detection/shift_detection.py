import logging
import os
from typing import Annotated, Optional
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import nibabel as nib
import json
import torch
import math, json

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# shift_detection
#


class shift_detection(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("shift_detection")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#shift_detection">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # shift_detection1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="shift_detection",
        sampleName="shift_detection1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "shift_detection1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="shift_detection1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="shift_detection1",
    )

    # shift_detection2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="shift_detection",
        sampleName="shift_detection2",
        thumbnailFileName=os.path.join(iconsPath, "shift_detection2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="shift_detection2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="shift_detection2",
    )


#
# shift_detectionParameterNode
#


@parameterNodeWrapper
class shift_detectionParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# shift_detectionWidget
#


class shift_detectionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/shift_detection.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = shift_detectionLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            print("firstVolumeNode:", firstVolumeNode)
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[shift_detectionParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# shift_detectionLogic
#


class shift_detectionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return shift_detectionParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Dosya uzantısını almak için
        file_path = inputVolume.GetStorageNode().GetFullNameFromFileName()
        logging.info(f"File: {file_path}\nProcessing started")
        
        start_process(file_path, output_file_path)



        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }

        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

#
# shift_detectionTest
#


class shift_detectionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_shift_detection1()

    def test_shift_detection1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("shift_detection1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = shift_detectionLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")



"""
    🛠️ This is the main script that will be used to process the input file and detect the midline shift.
"""


class ModelLoader:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)
    
    def predict(self, image):
        return self.model(image)

class FileManager:
    def __init__(self, input_path, output_path, points_folder):
        self.input_path = input_path
        self.output_path = output_path
        self.points_folder = points_folder
        os.makedirs(points_folder, exist_ok=True)
    
    def read_nii_file(self):
        img = nib.load(self.input_path)
        return img.get_fdata()

    def save_nii_file(self, image_data, header_data):
        img = nib.Nifti1Image(image_data, np.eye(4), dtype=np.uint8)
        img.header['pixdim'][4] = header_data['pixdim']
        img.header['xyzt_units'] = header_data['xyzt_units']
        nib.save(img, self.output_path)
        self.save_json_data(header_data['points_data'])

    def save_json_data(self, data):
        with open(f'{self.points_folder}/points.json', 'w') as f:
            json.dump(data, f, indent=4)

class DataProcessor:
    @staticmethod
    def extract_axial_slices(image_data):
        return image_data.transpose(2, 0, 1)
    
    @staticmethod
    def stack_slices(slices):
        return np.stack(slices, axis=0)

class SliceDrawer:
    def __init__(self, model, points_folder):
        self.model = model
        self.points_folder = points_folder
        self.id_number = 0
        self.data_points = self.initialize_data_points()
    
    def initialize_data_points(self):
        return {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Line",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": 2,
                    "controlPoints": [],
                    "measurements": [],
                    "display": self.initialize_display_properties()
                }
            ]
        }
    
    def initialize_display_properties(self):
        return {
            "visibility": True,
            "opacity": 1.0,
            "color": [0.4, 1.0, 1.0],
            "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
            "activeColor": [0.4, 1.0, 0.0],
            "propertiesLabelVisibility": True,
            "pointLabelsVisibility": False,
            "textScale": 3.0,
            "glyphType": "Sphere3D",
            "glyphScale": 3.0,
            "glyphSize": 5.0,
            "useGlyphScale": True,
            "sliceProjection": False,
            "sliceProjectionUseFiducialColor": True,
            "sliceProjectionOutlinedBehindSlicePlane": False,
            "sliceProjectionColor": [1.0, 1.0, 1.0],
            "sliceProjectionOpacity": 0.6,
            "lineThickness": 0.2,
            "lineColorFadingStart": 1.0,
            "lineColorFadingEnd": 10.0,
            "lineColorFadingSaturation": 1.0,
            "lineColorFadingHueOffset": 0.0,
            "handlesInteractive": False,
            "translationHandleVisibility": True,
            "rotationHandleVisibility": True,
            "scaleHandleVisibility": True,
            "interactionHandleScale": 3.0,
            "snapMode": "toVisibleSurface"
        }
    
    def draw_on_slice(self, idx, slice_img, len_slices):
        self.id_number += 1
        imgs = self.preprocess_image(slice_img)

        if idx <= len_slices / 4 or idx >= 3 * len_slices / 4:
            return self.rotate_image(imgs)

        results = self.model.predict(imgs)
        predictions = results.pred[0]
        boxes, scores, categories = predictions[:, :4], predictions[:, 4], predictions[:, 5]

        AF, PF, SP = self.detect_regions(categories, scores)
        if AF == -1 or PF == -1 or SP == -1:
            return self.rotate_image(imgs)

        brain_points, center_points = self.calculate_center_points(boxes, AF, PF, SP)
        self.data_points["markups"][0]["controlPoints"].extend(brain_points)
        
        self.draw_lines(center_points, brain_points)
        
        return self.rotate_image(imgs)

    def preprocess_image(self, slice_img):
        imgs = cv2.normalize(slice_img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if imgs.ndim == 2:
            imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2RGB)
        imgs = cv2.rotate(imgs, cv2.ROTATE_90_CLOCKWISE)
        imgs = cv2.resize(imgs, (160, 256))
        return imgs

    def rotate_image(self, image):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 0)
        return image

    def detect_regions(self, categories, scores):
        AF, PF, SP = -1, -1, -1
        for i, (category, score) in enumerate(zip(categories, scores)):
            if category == 0 and (AF == -1 or scores[AF] <= score) and score > 0.40:
                AF = i
            elif category == 1 and (PF == -1 or scores[PF] <= score) and score > 0.25:
                PF = i
            elif category == 2 and (SP == -1 or scores[SP] <= score) and score > 0.35:
                SP = i
        return AF, PF, SP

    def calculate_center_points(self, boxes, AF, PF, SP):
        brain_points = []
        center_points = []
        for i, box in enumerate([boxes[AF], boxes[PF], boxes[SP]]):
            x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            center_points.append((x_center, y_center))
            point_name = ["AF", "SP", "PF"][i]
            new_point = self.create_point(self.id_number, point_name, x_center, y_center)
            brain_points.append(new_point)
        return brain_points, center_points

    def create_point(self, id_number, label, x, y):
        return {
            "id": str(id_number),
            "label": f"F_{id_number}-{label}",
            "description": "",
            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
            "position": [float(id_number), float(x), float(y)],
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        }

    def draw_lines(self, center_points, brain_points):
        distances = squareform(pdist(center_points))
        max_indices = np.unravel_index(np.argmax(distances), distances.shape)
        start_point, end_point = center_points[max_indices[0]], center_points[max_indices[1]]
        
        a, b = end_point[1] - start_point[1], start_point[0] - end_point[0]
        c = end_point[0] * start_point[1] - start_point[0] * end_point[1]
        other_point = center_points[3 - max_indices[0] - max_indices[1]]
        
        distance = abs(a * other_point[0] + b * other_point[1] + c) / np.sqrt(a**2 + b**2)
        
        intersection_x = (b*(b*other_point[0] - a*other_point[1]) - a*c) / (a**2 + b**2)
        intersection_y = (a*(-b*other_point[0] + a*other_point[1]) - b*c) / (a**2 + b**2)
        intersection_point = (int(intersection_x), int(intersection_y))
        
        m, b = self.find_line_equation(start_point[0], start_point[1], end_point[0], end_point[1])
        x3, y3 = self.closest_point_on_line(other_point[0], other_point[1], m, b)
        
        self.draw_bresenham_line(start_point, end_point)
        self.draw_bresenham_line((int(x3), int(y3)), other_point)
        
        self.add_line_to_data(start_point, end_point)
        self.add_line_to_data((int(x3), int(y3)), other_point)
        
    def find_line_equation(self, x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def closest_point_on_line(self, x, y, m, b):
        m_perpendicular = -1 / m
        b_perpendicular = y - m_perpendicular * x
        intersection_x = (b_perpendicular - b) / (m - m_perpendicular)
        intersection_y = m * intersection_x + b
        return intersection_x, intersection_y

    def draw_bresenham_line(self, start, end):
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            self.points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def add_line_to_data(self, start, end):
        new_line = {
            "id": str(self.id_number),
            "label": f"Line-{self.id_number}",
            "description": "",
            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
            "position": [float(self.id_number), float(start[0]), float(start[1])],
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        }
        self.data_points["markups"][0]["controlPoints"].append(new_line)


if __name__ == "__main__":
    model_loader = ModelLoader("./model/model.pt")
    file_manager = FileManager("input_file.nii", "output_file.nii", "points_folder")
    
    image_data = file_manager.read_nii_file()
    slices = DataProcessor.extract_axial_slices(image_data)

    drawer = SliceDrawer(model_loader, 'points_folder')

    processed_slices = [drawer.draw_on_slice(idx, slice_img, len(slices)) for idx, slice_img in enumerate(slices)]

    stacked_slices = DataProcessor.stack_slices(processed_slices)

    header_data = {
        'pixdim': 3,
        'xyzt_units': 2,
        'points_data': drawer.data_points
    }

    file_manager.save_nii_file(stacked_slices, header_data)
    logging.info("Process completed successfully ✅")
    logging.info("Points data is saved as points_folder/points.json 📁")
