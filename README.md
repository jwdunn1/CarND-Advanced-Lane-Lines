## Advanced Lane Finding

Submission files:
<pre>
p4-SubmissionWriteup.pdf        Writeup
p4-SubmissionVideo.mp4          Annotated final video

calibrate.py                    Calibration code
pipeline.py                     Pipeline code



output_images directory:
1-overhead.mp4                  Warped overhead view
2-filters.mp4                   Binary filtered regions
3-fittedPoly.mp4                Fitted polynomials overdrawn
4-lane.mp4                      Lane painted with filter markers
5-FINAL.mp4                     Annotated final video
annotationTests                 Annotation tests
calibrationTests                Calibration tests
otherTests                      Miscellaneous tests
overheadTests                   Overhead space tests
polyfitTests                    Polyfit plots
thresholdTests                  Threshold tests
videoTests                      Video tests










</pre>

## Project Log
20170127 Begin project 4
- Organize directories and contents
- Camera calibration first steps
- Illustrator analysis // difficult to find a best
- Undistort/perspective transform produces good results apx 134x134 each 
- Start constructing processing pipeline: undistort images/video
- Idea: use maps to determine actual curvature
- Locate position on GoogleMaps.
- vid1 elements: drain at the start on the left, exit sign, 
- Use Illustrator/Excel to measure distance/scale
- Determine true curvature with margin of error to establish a known tolerance
- May be able to reference the horizon and determine direction/compass
- Every mobile device can contribute accuracy and cross-validation using gyro and accelerometers. For safety, not malice.
- Assumption: camera is not validated - wrote to Ryan Keenan to ask

20170128 Analysis cont'd
- Response from Ryan indicates validated
- Also realized that shape depends on fov and focal length
- 23in monitor positioned 10in away from eye produces correct effect in eye
- Search for videos 2 and 3 locations in maps
- vid2 elements: route 85/65/62 sign, Freemont Ave sign, Sunnyvale exits sign, arch shape in the distance at the end of the video, reverse exit sign, proximity to vid1, number of lanes, lane shift denotation
- vid2 is located on route 85NB from start of detour into side lane to end, crossing under the Homestead Road overpass. Distance is apx 1631.46ft or 497.27m in 16s.
- vid3 elements: 3 lines, tree type/height/density, 35mph, 15mph hairpin, shadow from signs, railing, bridge, cement structure after the last turn, proximity to vid1&2, motorcycle frequency, searching...
- vid3 is located on route 84SWB (La Honda Road) from a point after the bend beyond the Fox Hill Rd turn off to the 15mph bend up to the cement on the right. Distance is 1800ft or 548.73m in 47s. (2km SW of the western end of the Klystron Gallery at the SLAC National Accelerator Laboratory)

20170129 Begin image thresholding analysis
- Snap samples from the challenge videos
- Determine trapezoid for perspective transform
- Investigate color thresholding: pyimagesearch.com/2014/08/04/opencv-python-color-detection // found good results
- Note: when processing the video, BRG rather than RGB: invert the positions of the boundary numbers in the array.
- Warped the lines into overhead perspective
- Create overhead videos
- Vid2 needs additional filter: apply sobelx
- At t=14s, lane1: 282px, lane2: 267px, lane3: 250px L:200
- At t=18s, lane1: 289px, lane2: 283px, lane3: 281px L:180
- At t=20s, lane1: 291px, lane2: 300px, lane3: 328px L:181

20170130 Scale needs to be determined
- Capture section of straight freeway to determine ratio: 13.86% or 111/801
- Capture section of min radius to check: match!
- Plan is to capture three points along left line to determine curvature (top, mid, bot) Like: easycalculation.com/analytical/three-points-circle-equation.php
- For the fill, need to fit curve/poly for left and right lines.
- Measurements with Google Maps conclude the white pavement markings on 280 are 4.5m 14.7638ft (segments) and 9m 29.5276ft (gaps)
- dot.ca.gov/trafficops/camutcd/docs/TMChapter6.pdf
indicates 48feet(14.64m) dot-to-dot w/ 12feet lines
found specs for left, right, center
: "The edgeline should be placed 50mm in from the edge of traveled way, approximately 3.6 m from the laneline or centerline on highway mainlines"
- 68px=12ft=3.6576m lane width (basis of scaling)
- 56.67px=10ft lane width in vid2     12ft lane width in vid3
- First 
- Use bridge as trapezoidal measure: straight, known
67.84m length, 12ft width
- Attempt adj trapez and compare to orig bridge
- When annotating, fade into distance as uncertainty increases.
- Curvature error with alt trapez: 1782ft (should be 3000ft)

20170131 Continue resolving scale issue
- Create routine for radius from three points (intmath.com/applications-differentiation/8-radius-curvature.php)
- Work in Illustrator to determine vanishing point and trap
- Average length of sedan: 4.5m (450cm) or 177.2in or 14.76ft
- Spec for reflector-to-reflector placement is 14.64m or 48ft
- Draw parallel lanes at scale using base image
- Determine source/destination points
- First too long, second too short, third is tolerable
- Next step is to determine lines.

20170201 Determine lines
- Lane Detection and Tracking with MATLAB: youtu.be/SFqAAseL_1g
- PennDOT "Line Painting" ca. 1968: youtu.be/S1Lu6NpZTXc
- What are road markings: youtu.be/_KaHbbVxJWE
- Extant: cv2.minEnclosingCircle(points)
- Investigate code on polyfit
- May be able to run a hist at top, mid, bot of image rather than using the stack approach discussed in class. Perhaps a stack of 100px.
- Just realized all efforts for precision measurement were performed using original (distorted) video.
- Scale of final 59px to 12ft (3.6576m)
- 59px to 12ft in x, but y appears long.
- Revisit dest rectangle in Illustrator: determine 59x472px lane
- Adjust filter to thresholded S channel

20170202 Histogram
- Find starting point: localized hist
- Research OpenCV on GPU
- Explore tuning of filters
- Idea: localized regions of interest LRI
- Establish confidence measure and target

20170203 Straight segment
- Idea: dl patches
- Began the drawPoly()
- Explore filters in Picasa and Photoshop for shadow areas
- From this reparm the sobel and YW filters
- Dynamic adjustment to upper LRI sizes on rate of curvature
- Outlier rejection if confidence is established by partner
	Alt.: mutcd.fhwa.dot.gov/htm/2003r1/part3/part3a.htm

20170204 Planning in Illustrator
- Dynamic scan
- Bounce/pitch is a problem: it can widen the lanes further out

20170205 Rotate clockwise
- Remeasure scale and produce horizontal views
- sobely
- Step 1, locate first left primary line (yellow)
- Step 2, confirm if possible secondary (white) 3/4 failure
Adjust lane width
- Step 3, if high confidence, shift lane
- Step 4, if curve, calc y at x:303, locate next left primary
- Step 5, if confident and non-linear, fit curve with mirror
- Step 6, 
- Exploring patch filtering and lane alignment principles
- Perhaps use np.var for ranking? or ideal range? How is a bad histogram identified?
- Maybe start with lines out to 303 in x, see how that runs, get the pipeline operating and annotating straight video
- Method to follow lines:
	1, Acquire L-patch1, determine y from hist
	2, use default lane width and
	...
	Thinner, wider patches may work better for vid3

20170206 First annotation
- Rework scaling again
- Add overlay on straight segment
- Track lines using patches
- MakeGradient

20170207
- Add additional rules to average lane width and reasonable angle
- Refactor LR code

20170208
- Draw semi-opaque box in HUD area using PIL
- Explore font render quality // Open Sans
- Add center points code and polylines to draw
- Todos:
	1. fit splines
	2. cascading thesholds in shade (for yellow also)
	3. write up
	- Clip 640 frames for right curve
	- Fix distant threshold using eqhist on right

20170209
- Outliers set 30px caused problems in shadow area // set 10
- Add lane drift computation and display in cm
- Use centerline as curve measure
- Debug averaging code
- Change display code to PIL
- Fix distant threshold using lower thesh on right
- Repair averaging (added numpy.copy method), lower to 29 frames
- Update interface elements and negative centerline

20170210 Polynomial
- Plot overlay of polynomial on line data points
- Compute polynomial using averaged centerline points
- Display centerline using fitted polynomial
- Compute radius using fitted polynomial
- Explore dynamic queue length control...holding turns
- May need additional investigation on right line thresholding
- Outline for writeup:
	-	Camera calibration and distortion correction
	-	Perspective transform
	-	Method to identify lane lines
	-	Center line average polynomial fitting
	-	Curvature and lane drift
	-	Annotation
	-	Discussion
- Lower threshold from 127 to 64 on left line
- Attempt eqhist on primary // too squirrely
- Iterative refinement: install white and yellow filter + rule

20170211 Refinement
- Add patch diags view
- Computed average lane width is 160.66
- Reassess the dynamic queue approach, Idea: perhaps a multiqueue like on the last project: several lengths, use best fitting.
- Found bug: FilterW called on left rather than FilterY
- Lower the threshold on FilterY // still not working
- Add queue to right line width values to avg in blank areas
- Found bug: Needed to flip red and blue values in FilterY
- Width of lines is spec'd by CalDot to be 100mm or 

20170212 Refactor
- Create NQ, Executive, Model, and Line classes
- Rebuild code using classes
- Use avg on left side; lengthen center and width queues

20170213 
- Display lane edges using fitted curves
- Rotate diagnostics display of patches
- Still wobbling, shorten queues // OK
- Restructure drawData routine
- Bug: need to output undistorted images in video
- Attempt prefilter on challenge video
- Need additional rule to gate lane width
- Ideas for improvement: lighting sense, dynamic patch width and number, initial width, noise filter

20170214 Brainstorm
- In the advanced challenge:
    - Follow a fitted polynomial/spline to obtain and map a diagonal patch into a rectilinear.
    - Process left/right of patch if failed
    - GPS map data compare
- Dynamic cascading color threshold carries from the prior patch
- Determine overall level of light to decide filtering
- Convolve hist to find center
- Investigate patch lightness corrections

20170215 Fix derailment at frame 1060
- Analysis reveals camera/vehicle bounce (narrowing lines)
- Attempt remedies such as centerline offset, widen outlier tolerance, and slight offset to lane display.
- Begin writeup

20170216 Better word for "patch"...
	target, isograph, fragment, monitor, geograph, geogram, frame, sample, cell, loci, overlay, locus, odometry, volumetric grid, sensing panel, scanning panel, plate, filter region, region of interest, patch network, zone, compartment, partition, detection zone, discovery zone, identify zone, selection zone, network for discovery, interval, scan zone, sample point, survey, filterable region, selection, --> "filter"

20170217 Code clean up

20170218 Formatting in Word
- Problem with producing high-quality PDFs from Word

20170219 Move to Adobe InDesign
- Refine layout and writeup
- Package files for submission
- End project 4