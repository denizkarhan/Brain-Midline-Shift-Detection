import logging
import os
from typing import Annotated, Optional
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import nibabel as nib
import json
import torch
import math

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
        self.parent.title = _("Shift Detection")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Deniz Developer"]
        self.parent.helpText = _("""
This module detects brain shift using YOLOv5 model.
""")
        self.parent.acknowledgementText = _("""
Brain shift detection module using deep learning.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # shift_detection1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="shift_detection",
        sampleName="shift_detection1",
        thumbnailFileName=os.path.join(iconsPath, "shift_detection1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="shift_detection1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="shift_detection1",
    )

    # shift_detection2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="shift_detection",
        sampleName="shift_detection2",
        thumbnailFileName=os.path.join(iconsPath, "shift_detection2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="shift_detection2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="shift_detection2",
    )

#
# shift_detectionParameterNode
#

@parameterNodeWrapper
class shift_detectionParameterNode:
    """
    The parameters needed by module.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode

#
# shift_detectionWidget
#

class shift_detectionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class"""

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/shift_detection.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = shift_detectionLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
                
        # Create default output volume if not exists
        if not self._parameterNode.thresholdedVolume:
            outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ShiftDetection_Output")
            if outputVolume:
                self._parameterNode.thresholdedVolume = outputVolume

    def setParameterNode(self, inputParameterNode: Optional[shift_detectionParameterNode]) -> None:
        """Set and observe parameter node."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        
        self._parameterNode = inputParameterNode
        
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Start brain shift detection")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks Apply button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Get input volume from UI
            inputVolume = self.ui.inputSelector.currentNode()
            outputVolume = self.ui.outputSelector.currentNode()
            
            if not inputVolume:
                slicer.util.errorDisplay("Please select an input volume.")
                return
                
            if not outputVolume:
                slicer.util.errorDisplay("Please select an output volume.")
                return
            
            # Process the volume
            self.logic.process(inputVolume, outputVolume)
            
            # Handle inverted output if selected
            if self.ui.invertedOutputSelector.currentNode():
                invertedVolume = self.ui.invertedOutputSelector.currentNode()
                # You can add additional processing for inverted volume here if needed
                logging.info("Inverted output volume selected")

#
# shift_detectionLogic
#

class shift_detectionLogic(ScriptedLoadableModuleLogic):
    """This class implements the actual computation."""

    def __init__(self) -> None:
        """Called when the logic class is instantiated."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.model = None
        self.idNumber = 0
        self.dataPoints = None
        self.initialize_model()
        self.initialize_data_structure()

    def initialize_model(self):
        """Initialize YOLOv5 model"""
        try:
            model_path = '/Users/deniz/shift_detection/shift_detection/model/model.pt'
            if os.path.exists(model_path):
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)
                logging.info("YOLOv5 model loaded successfully")
            else:
                logging.error(f"Model file not found at: {model_path}")
                self.model = None
        except Exception as e:
            logging.error(f"Failed to load YOLOv5 model: {e}")
            self.model = None

    def initialize_data_structure(self):
        """Initialize data structure for points"""
        self.dataPoints = {
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
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": [0.4, 1.0, 1.0],
                        "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": True,
                        "pointLabelsVisibility": False,
                        "textScale": 1.5,
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
                }
            ]
        }

    def getParameterNode(self):
        return shift_detectionParameterNode(super().getParameterNode())

    def process(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode = None) -> None:
        """Run the processing algorithm."""
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        import time
        startTime = time.time()
        logging.info("Processing started")

        # Get file path
        storageNode = inputVolume.GetStorageNode()
        if not storageNode:
            raise ValueError("Input volume has no storage node")
        
        file_path = storageNode.GetFullNameFromFileName()
        if not file_path:
            raise ValueError("Cannot get file path from input volume")
        
        logging.info(f"Processing file: {file_path}")

        # Set output paths
        output_file_path = '/Users/deniz/shift_detection/shift_detection/resultMLS/result.nii'
        points_folder_path = '/Users/deniz/shift_detection/shift_detection/resultMLS/pointsAndLines'
        
        # Create output directories
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        os.makedirs(points_folder_path, exist_ok=True)

        # Reset counters
        self.idNumber = 0
        self.initialize_data_structure()

        # Process the file
        self.start_process(file_path, output_file_path, points_folder_path)
        
        # If output volume is provided, load the result into it
        if outputVolume:
            try:
                # Load the processed volume back into Slicer
                loadedVolume = slicer.util.loadVolume(output_file_path)
                if loadedVolume:
                    # Copy the data to the output volume
                    outputVolume.SetAndObserveImageData(loadedVolume.GetImageData())
                    outputVolume.SetOrigin(loadedVolume.GetOrigin())
                    outputVolume.SetSpacing(loadedVolume.GetSpacing())
                    outputVolume.SetIJKToRASDirections(loadedVolume.GetIJKToRASDirections())
                    
                    # Remove the temporarily loaded volume
                    slicer.mrmlScene.RemoveNode(loadedVolume)
                    
                    # Update display
                    slicer.util.setSliceViewerLayers(background=outputVolume)
            except Exception as e:
                logging.warning(f"Could not load result into output volume: {e}")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def draw_on_slice(self, idx, slice_img, len_slices, points_folder_path):
        """Process individual slice with YOLOv5 detection"""
        if self.model is None:
            logging.warning("Model not loaded, skipping slice processing")
            return slice_img

        self.idNumber += 1
        
        # Normalize image
        imgs = cv2.normalize(slice_img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if imgs.ndim == 2:
            imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2RGB)

        imgs = cv2.rotate(imgs, cv2.ROTATE_90_CLOCKWISE)

        # Resize to target shape
        target_shape = (256, 160, 3)
        imgs = cv2.resize(imgs, (target_shape[1], target_shape[0]))

        # Skip processing for edge slices
        if idx <= len_slices / 4 or idx >= 3 * len_slices / 4:
            imgs = cv2.rotate(imgs, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgs = cv2.flip(imgs, 0)
            return imgs

        # Run YOLOv5 detection
        try:
            results = self.model(imgs)
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            # Find best detections for each category
            AF, PF, SP = -1, -1, -1
            for i, (category, score) in enumerate(zip(categories, scores)):
                if category == 0 and (AF == -1 or scores[AF] <= score):
                    if score > 0.40: AF = i
                elif category == 1 and (PF == -1 or scores[PF] <= score):
                    if score > 0.25: PF = i
                elif category == 2 and (SP == -1 or scores[SP] <= score):
                    if score > 0.35: SP = i

            # Check if all required points are detected
            if AF == -1 or PF == -1 or SP == -1:
                imgs = cv2.rotate(imgs, cv2.ROTATE_90_COUNTERCLOCKWISE)
                imgs = cv2.flip(imgs, 0)
                return imgs

            logging.info(f"{self.idNumber}-> AF:{scores[AF]:.3f} | SP:{scores[SP]:.3f} | PF:{scores[PF]:.3f}")

            # Process detected points
            boxes2 = [boxes[AF], boxes[PF], boxes[SP]]
            brainPoints, center_points, arr = [], [], []
            
            for a, box in enumerate(boxes2):
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                center_points.append((x_center, y_center))
                arr.append([x_center, y_center])
                
                pointName = "AF" if a == 0 else ("SP" if a == 1 else "PF")
                new_point = {
                    "id": str(self.idNumber),
                    "label": f"F_{self.idNumber}-{pointName}",
                    "description": "",
                    "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                    "position": [float(self.idNumber), float(x_center), float(y_center)],
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": True,
                    "locked": False,
                    "visibility": True,
                    "positionStatus": "defined"
                }
                self.dataPoints["markups"][0]["controlPoints"].append(new_point)
                brainPoints.append(new_point)

            # Calculate geometric relationships
            distances = squareform(pdist(center_points))
            max_indices = np.unravel_index(np.argmax(distances), distances.shape)

            # Find line equation and closest point
            m, b = self.find_line_equation(arr[0][0], arr[0][1], arr[1][0], arr[1][1])
            x3, y3 = self.closest_point_on_line(arr[2][0], arr[2][1], m, b)

            # Create additional points for measurements
            new_point4 = {
                "id": str(self.idNumber),
                "label": "",
                "description": "",
                "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                "position": [float(self.idNumber), x3, y3],
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined"
            }
            
            new_point5 = {
                "id": str(self.idNumber),
                "label": "",
                "description": "",
                "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                "position": [float(self.idNumber), float(arr[2][0]), float(arr[2][1])],
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined"
            }

            # Save line measurements
            self.line2Json(brainPoints[0], brainPoints[1], self.idNumber, 1, points_folder_path)
            self.line2Json(brainPoints[1], brainPoints[2], self.idNumber, 2, points_folder_path)
            self.line2Json(brainPoints[0], brainPoints[2], self.idNumber, 3, points_folder_path)
            self.line2Json(new_point4, new_point5, self.idNumber, 4, points_folder_path)

        except Exception as e:
            logging.error(f"Error processing slice {idx}: {e}")

        # Return processed image
        imgs = cv2.rotate(imgs, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgs = cv2.flip(imgs, 0)
        return imgs

    def read_nii_file(self, file_path):
        """Read NIfTI file"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not file_path.endswith('.nii'):
            raise ValueError(f"File is not a valid NIfTI file: {file_path}")
        
        img = nib.load(file_path)
        image_data = img.get_fdata()
        if image_data is None:
            raise ValueError(f"Failed to load NIfTI file: {file_path}")
        return image_data

    def extract_axial_slices(self, image_data):
        """Extract axial slices from 3D volume"""
        return image_data.transpose(2, 0, 1)

    def stack_slices(self, slices):
        """Stack processed slices back into 3D volume"""
        return np.stack(slices, axis=0)

    def save_nii_file(self, file_path, image_data, points_folder_path):
        """Save processed volume and points"""
        img = nib.Nifti1Image(image_data, np.eye(4), dtype=np.uint8)
        img.header['pixdim'][4] = 3
        img.header['xyzt_units'] = 2
        nib.save(img, file_path)
        
        # Save points data
        if self.dataPoints:
            with open(f'{points_folder_path}/points.json', 'w') as f:
                json.dump(self.dataPoints, f, indent=4)
        else:
            logging.warning("dataPoints is not defined!")

    def line2Json(self, firstPoint, secondPoint, id_Number, num, points_folder_path):
        """Create JSON file for line measurements"""
        newDataLines = {
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
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": [0.4, 1.0, 1.0],
                        "selectedColor": [0.0, 0.56, 0.11] if num == 1 else ([1.0, 0.14, 0.14] if num == 4 else [0.34, 0.41, 0.84]),
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": True,
                        "pointLabelsVisibility": False,
                        "textScale": 1.25 if num == 4 else 0.0,
                        "glyphType": "Sphere3D",
                        "glyphScale": 2.40 if num == 1 else (3.0 if num == 4 else 1.40),
                        "glyphSize": 5.0,
                        "useGlyphScale": True,
                        "sliceProjection": False,
                        "sliceProjectionUseFiducialColor": True,
                        "sliceProjectionOutlinedBehindSlicePlane": False,
                        "sliceProjectionColor": [1.0, 1.0, 1.0],
                        "sliceProjectionOpacity": 0.6,
                        "lineThickness": 1.0 if num == 1 else 0.6,
                        "lineColorFadingStart": 1.0,
                        "lineColorFadingEnd": 10.0,
                        "lineColorFadingSaturation": 1.0,
                        "lineColorFadingHueOffset": 0.0,
                        "handlesInteractive": False,
                        "translationHandleVisibility": True,
                        "rotationHandleVisibility": True,
                        "scaleHandleVisibility": True,
                        "interactionHandleScale": 2.0 if num == 1 else 3.0,
                        "snapMode": "toVisibleSurface"
                    }
                }
            ]
        }
        
        # Calculate line length
        lineLen = math.sqrt(((firstPoint['position'][2] - secondPoint['position'][2]) ** 2) + 
                           ((firstPoint['position'][1] - secondPoint['position'][1]) ** 2))
        
        measurements = {
            "name": "length",
            "enabled": True,
            "value": float(lineLen),
            "units": "mm",
            "printFormat": "%-#4.4gmm"
        }
        
        newDataLines["markups"][0]["controlPoints"].append(firstPoint)
        newDataLines["markups"][0]["controlPoints"].append(secondPoint)
        newDataLines["markups"][0]["measurements"].append(measurements)

        # Save to file
        with open(f'{points_folder_path}/file_{id_Number}_{num}.json', 'w') as f:
            json.dump(newDataLines, f, indent=4)

    def distance(self, x1, y1, x2, y2):
        """Calculate distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def point_to_line_distance(self, x0, y0, x1, y1, x2, y2):
        """Calculate distance from point to line"""
        return abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1) / self.distance(x1, y1, x2, y2)

    def find_line_equation(self, x1, y1, x2, y2):
        """Find line equation parameters"""
        if x2 - x1 == 0:
            m = float('inf')
        else:
            m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def closest_point_on_line(self, x, y, m, b):
        """Find closest point on line to given point"""
        if m == float('inf'):
            # Vertical line case
            return x, y
        
        x3 = float((x + m*y - m*b) / (m**2 + 1))
        y3 = float((m*x + (m**2)*y + b) / (m**2 + 1))
        return x3, y3

    def start_process(self, input_file_path, output_file_path, points_folder_path):
        """Main processing function"""
        if not os.path.isfile(input_file_path):
            raise FileNotFoundError(f"Input file does not exist: {input_file_path}")
        if not input_file_path.endswith('.nii'):
            raise ValueError(f"Input file is not a valid NIfTI file: {input_file_path}")
        
        # Load and process the NIfTI file
        image_data = self.read_nii_file(input_file_path)
        logging.info(f"Loaded image data with shape: {image_data.shape}")
        
        # Extract axial slices
        slices = self.extract_axial_slices(image_data)
        
        # Process each slice
        modified_slices = []
        for idx, slice_data in enumerate(slices):
            processed_slice = self.draw_on_slice(idx, slice_data, len(slices), points_folder_path)
            modified_slices.append(processed_slice)
        
        # Stack slices back into 3D volume
        modified_image_data = self.stack_slices(modified_slices)
        
        # Save the processed volume and points
        self.save_nii_file(output_file_path, modified_image_data, points_folder_path)
        logging.info("Processing completed successfully")

#
# shift_detectionTest
#

class shift_detectionTest(ScriptedLoadableModuleTest):
    """Test case for the scripted module."""

    def setUp(self):
        """Reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_shift_detection1()

    def test_shift_detection1(self):
        """Test the module functionality."""
        self.delayDisplay("Starting the test")

        # Get/create input data
        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample("shift_detection1")
        self.delayDisplay("Loaded test data set")

        # Test the module logic
        logic = shift_detectionLogic()
        
        # Create output volume
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestOutput")
        
        # Basic test
        try:
            logic.process(inputVolume, outputVolume)
            self.delayDisplay("Test passed")
        except Exception as e:
            self.delayDisplay(f"Test failed: {e}")
            raise e
