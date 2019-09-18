# -*- coding: utf-8 -*-


import argparse
import itertools
import json
import operator
import sys
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize
import scipy.stats
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate

from dataset_info import *
from loading import *
from utils import *

matplotlib.rcParams['text.latex.unicode'] = True
font = {'size': 13}
matplotlib.rc('font', **font)


# %%
DEFAULT_CONFIG = "config.py"
MEASUREMENT_MODES = set(["median", "full"])
USE_PERCENTIL = 95.0
WIDTH = 1920
HEIGHT = 1080
SAFE_BORDER_OFFSET = 10
MAX_TIME_DIFF = 0.2  # in seconds


# %%
"""
##############################################################
####################### SPEED and MATCHING ###################
##############################################################
"""

"""
For each car compute speeds between lines
"""


def calculateSpeeds(sessionId, recordingId, data, gtData, system):
    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(data["camera_calibration"]["vp1"],
                                                                   data["camera_calibration"]["vp2"],
                                                                   data["camera_calibration"]["pp"])

    def pTran(p): return os.path.join(
        getPathForRecording(sessionId, recordingId), p)
    lines = gtData["measurementLines"]
    laneDivLines = gtData["laneDivLines"]
    fps = gtData["fps"]
    errors = 0
    for car in data["cars"]:
        intersectionData = [getCarTimeAndSpatialIntersection(
            l, car["posX"], car["posY"], car["frames"]) for l in lines]
        startSpatial, startFrame, startLineSpatial = intersectionData[-1]
        endSpatial, endFrame, endLineSpatial = intersectionData[0]

        # test on poorly approximated intersections
        testMat = np.concatenate((endSpatial, startSpatial,
                                  [startFrame, endFrame]))
        if not (np.any(np.isnan(testMat)) or
                np.any(np.isinf(testMat)) or
                getLaneForPoint(endSpatial, laneDivLines) is None or
                getLaneForPoint(startSpatial, laneDivLines) is None):

            elapsedTime = abs(endFrame - startFrame) / fps
            def projector(p): return getWorldCoordinagesOnRoadPlane(
                p, focal, roadPlane, pp)
            passedDistance = data["camera_calibration"]["scale"] * \
                np.linalg.norm(projector(startSpatial) - projector(endSpatial))
            car["speed"] = passedDistance / elapsedTime * 3.6
            car["laneIndex"] = getLaneForPoint(endSpatial, laneDivLines)
            car["timeIntersectionLast"] = endFrame / fps

            car["timeIntersectionFirst"] = startFrame / fps
            car["spatialIntersectionFirst"] = startSpatial
            car["spatialIntersectionLast"] = endSpatial

            points = list(map(lambda x, y: np.array(
                [x, y, 1]), car["posX"], car["posY"]))
            points = list(map(projector, points))
            frames = car["frames"]
            perFrameSpeeds = []
            pointsOffset = 5
            for i in range(len(points) - pointsOffset):
                passedDistance = data["camera_calibration"]["scale"] * \
                    np.linalg.norm(points[i] - points[i + pointsOffset])
                elapsedTime = abs(frames[i] - frames[i + pointsOffset]) / fps
                perFrameSpeeds.append(passedDistance / elapsedTime * 3.6)
            car["medianSpeed"] = np.median(perFrameSpeeds)

        else:
            errors += 1
            if SHOW_ERRORS:
                img = scipy.misc.imread(pTran("screen.png"))
                scale = float(17.5) / WIDTH
                plt.figure(
                    figsize=(img.shape[1] * scale, img.shape[0] * scale))
                plt.imshow(img)

                for l in lines:
                    drawLinePyplot(l, color="yellow", ls="--", linewidth=2)
                drawLinePyplot(startLineSpatial,  color="red",
                               linewidth=2, label="approximation to first line")
                drawLinePyplot(endLineSpatial,  color="blue",
                               linewidth=2, label="approximation to last line")
                plt.plot(car["posX"], car["posY"], "o",
                         color="black", markersize="4", label="trajectory")
                plt.legend()
                plt.title("ERROR approximating car: %s %s %s %d" %
                          (system, sessionId, recordingId, car["id"]))
                plt.show()

    return {"roadPlane": roadPlane, "focal": focal, "pp": pp}, errors


"""
finds matches between gt and observed vehicles
"""


def computeMatches(gtData, data, sessionId, recordingId, systemId):
    matches = []
    for gtCar in gtData["cars"]:
        filtered = [i for i in data["cars"]
                    if "laneIndex" in i and i["laneIndex"] in gtCar["laneIndex"]]
        gtTimeIntersection = gtCar["intersections"][-1]["videoTime"]
        filtered.sort(key=lambda i: abs(
            i["timeIntersectionLast"] - gtTimeIntersection))
        if len(filtered) > 0 and abs(filtered[0]["timeIntersectionLast"] - gtTimeIntersection) < MAX_TIME_DIFF:
            gtSpeed = gtCar["speed"]
            if MEASUREMENT_MODE == "full":
                speed = filtered[0]["speed"]
            elif MEASUREMENT_MODE == "median":
                speed = filtered[0]["medianSpeed"]
            else:
                assert False, "invalid measurement mode"

            matches.append({"matched": True,
                            "gtSpeed": gtSpeed,
                            "speed": speed,
                            "gtId": gtCar["carId"],
                            "valid": gtCar["valid"],
                            "matchedId": filtered[0]["id"],
                            "lastVideoTime": gtCar["intersections"][-1]["videoTime"],
                            "firstVideoTime":  gtCar["intersections"][0]["videoTime"]})

            diff = abs(gtSpeed - speed)
            if systemId in SHOW_BAD_FOR_SYSTEMS and diff > SHOW_BAD_THRESHOLD and gtCar["valid"]:
                car = filtered[0]
                def pTran(p): return os.path.join(
                    getPathForRecording(sessionId, recordingId), p)
                img = scipy.misc.imread(pTran("screen.png"))
                lines = gtData["measurementLines"]

                scale = float(17.5) / WIDTH
                plt.figure(
                    figsize=(img.shape[1] * scale, img.shape[0] * scale))
                plt.imshow(img)

                for l in lines:
                    drawLinePyplot(l, color="yellow", ls="--", linewidth=2)
                plt.plot(car["posX"], car["posY"], "o",
                         color="black", markersize="4", label="trajectory")
                plt.title("%s, %s %s, gt: %.2f meas:%.2f diff: %.2f, valid: %d, carId: %d, gtCarId: %d, videoTime: %f, gtLane: %d" % (systemId, sessionId, recordingId,
                                                                                                                                      gtCar["speed"], filtered[
                                                                                                                                          0]["speed"],
                                                                                                                                      gtCar["speed"] - filtered[0]["speed"],                                                                                       gtCar[
                                                                                                                                          "valid"], filtered[0]["id"], gtCar["carId"],
                                                                                                                                      gtCar["intersections"][-1]["videoTime"], list(gtCar["laneIndex"])[0]))
                plt.show()
        else:
            matches.append({"matched": False, "valid": gtCar["valid"]})
    return matches


# %%
"""
##############################################################
####################### HELP FUNCTIONS #######################
##############################################################
"""


def isCarBetweenLines(car, lines):
    assert len(car["frames"]) == len(car["posX"])
    assert len(car["frames"]) == len(car["posY"])
    betweenLines = 0
    for x, y in zip(car["posX"], car["posY"]):
        p = [x, y, 1]
        if x >= 0 and x <= WIDTH and\
                y >= 0 and y <= HEIGHT and\
                isPointBetweenLines(p, lines[0], lines[-1]):
            betweenLines += 1
        if betweenLines >= 6:
            return True
    return False


def filterPointsForCar(car):
    newXs = []
    newYs = []
    newFrames = []
    for i in range(len(car["posX"])):
        x, y, frame = car["posX"][i], car["posY"][i], car["frames"][i]
        if x > SAFE_BORDER_OFFSET and x < WIDTH - SAFE_BORDER_OFFSET\
                and y > SAFE_BORDER_OFFSET and y < HEIGHT - SAFE_BORDER_OFFSET:
            newXs.append(x)
            newYs.append(y)
            newFrames.append(frame)
    car["posX"] = newXs
    car["posY"] = newYs
    car["frames"] = newFrames


def filterInvalidatedLanesCars(car, laneDivLines, invalidLanes):
    if len(invalidLanes) == 0:
        return True
    for i in range(len(car["posX"])):
        laneId = getLaneForPoint(
            [car["posX"][i], car["posY"][i], 1], laneDivLines)
        if laneId in invalidLanes:
            return False
    return True


def prefilterData(data, gtData):
    fps = gtData["fps"]
    maxLineIntersection = max(
        [i["intersections"][-1]["videoTime"] for i in gtData["cars"]])
    # remove cars which were observed after last ground truth car
    data["cars"] = [i for i in data["cars"] if i["frames"][0] / fps <
                    maxLineIntersection]
    # remove cars which have only a number as positions
    data["cars"] = [c for c in data["cars"] if isinstance(c["posX"], list)]
    # remove points on edges
    for car in data["cars"]:
        filterPointsForCar(car)
    # remove cars which were observed on less then 5 frames
    data["cars"] = [i for i in data["cars"] if len(i["frames"]) > 5]
    # remove cars which are outside of predefined lanes dividing lines
    data["cars"] = [i for i in data["cars"] if list(map(lambda x, y:
                                                        getLaneForPoint(np.array([x, y, 1]), gtData["laneDivLines"]), i["posX"], i["posY"])).count(None) == 0]
    # remove cars which are for a moment in invalidated lanes
    data["cars"] = [car for car in data["cars"] if filterInvalidatedLanesCars(
        car, gtData["laneDivLines"], gtData["invalidLanes"])]
    # remove cars which are not between measurement lines
    data["cars"] = [car for car in data["cars"] if isCarBetweenLines(
        car, gtData["measurementLines"])]
    # remove contours from data (if they are present)
    for car in data["cars"]:
        if "contours" in car:
            del car["contours"]


"""
Computes intersection of car's track with a line in space and time
"""


def getCarTimeAndSpatialIntersection(line, posX, posY, frames, around=6):
    points = list(map(lambda x, y, f: (
        np.array([x, y, 1]), f), posX, posY, frames))
    points.sort(key=lambda i: pointToLineDistance(i[0], line))
    def getItems(getter): return list(
        map(getter, points[:min(around, len(points))]))
    spatialLineSlope, spatialLineIntercept = scipy.stats.linregress(getItems(lambda i: i[0][0]),
                                                                    getItems(lambda i: i[0][1]))[:2]
    spatialLine = np.array([spatialLineSlope, -1, spatialLineIntercept])
    spatialIntersection = np.cross(spatialLine, line)
    spatialIntersection = spatialIntersection / spatialIntersection[-1]

    frameNumbers = getItems(lambda i: i[1])
    normalizationPoint = np.cross(spatialLine, [0, 1, 0])
    normalizationPoint = normalizationPoint / normalizationPoint[-1]
    pointsProjected = [pointToLineProjection(
        spatialLine, p[0]) for p in points[:min(around, len(points))]]
    pointsDists = [np.linalg.norm(
        normalizationPoint - p) for p in pointsProjected]
    temporalLineSlope, temporalLineIntercept = scipy.stats.linregress(
        pointsDists, frameNumbers)[:2]
    temporalIntersection = temporalLineSlope * \
        np.linalg.norm(spatialIntersection -
                       normalizationPoint) + temporalLineIntercept

    return spatialIntersection, temporalIntersection, spatialLine


# %%
"""
##############################################################
###################### ERROR COMPUTATION #####################
##############################################################
"""


def computeFalsePositives(gtData, matches, data):
    maxLineIntersection = max(
        [i["intersections"][-1]["videoTime"] for i in gtData["cars"]])
    minLineIntersection = min(
        [i["intersections"][-1]["videoTime"] for i in gtData["cars"]])
    matchedCars = set([i["matchedId"]
                       for i in [it for it in matches if it["matched"]]])
    filtered = [i for i in data["cars"] if "timeIntersectionLast" in i
                and minLineIntersection <= i["timeIntersectionLast"]
                and i["timeIntersectionLast"] < maxLineIntersection]
    return len([i for i in filtered if i["id"] not in matchedCars])


def computeErrors(matches, errorType="absolute"):
    filtered = [i for i in matches if i["matched"] and i["valid"]]
    def errorFn(gt, measured): return abs(gt - measured)
    if errorType == "relative":
        def errorFn(gt, measured): return abs(gt - measured) / gt * 100
    elif errorType == "absoluteSign":
        def errorFn(gt, measured): return measured - gt
    return [errorFn(i["gtSpeed"], i["speed"]) for i in filtered]


def computeRecall(matches):
    valid = [i for i in matches if i["valid"]]
    return len([i for i in valid if i["matched"]]) / len(valid)


# %%
"""
##############################################################
####################### ERROR TABLES #########################
##############################################################
"""


def showErrorStats(systemsData):
    print("SPEED ERROR STATS")
    outputAbs = {}
    outputRel = {}
    outputSign = {}
    results = {}
    for k in RUN_FOR_SYSTEMS:
        results[k] = {}
        data = systemsData[k]
        errorsRel = []
        errorsAbs = []
        errorsSign = []
        tabulateData = []
        for videoId in RUN_FOR_VIDEOS:
            videoData = data[videoId]
            videoErrorsAbs = computeErrors(videoData["matches"], "absolute")
            videoErrorsRel = computeErrors(videoData["matches"], "relative")
            videoErrorsSign = computeErrors(
                videoData["matches"], "absoluteSign")
            results[k][videoId] = {"abs": getErrorsStats(
                videoErrorsAbs), "rel": getErrorsStats(videoErrorsRel)}
            errorsRel += videoErrorsRel
            errorsAbs += videoErrorsAbs
            errorsSign += videoErrorsSign
            tabulateData.append((("%s %s" % videoId).replace("session", "s"), len(videoErrorsAbs),
                                 np.mean(videoErrorsAbs), np.median(videoErrorsAbs), np.percentile(
                                     videoErrorsAbs, USE_PERCENTIL), np.max(videoErrorsAbs),
                                 np.mean(videoErrorsRel), np.median(videoErrorsRel), np.percentile(videoErrorsRel, USE_PERCENTIL), np.max(videoErrorsRel)))
        outputAbs[k] = errorsAbs
        outputRel[k] = errorsRel
        outputSign[k] = errorsSign
        tabulateData.append(("TOTAL", len(errorsAbs), np.mean(errorsAbs), np.median(errorsAbs), np.percentile(errorsAbs, USE_PERCENTIL), np.max(
            errorsAbs), np.mean(errorsRel), np.median(errorsRel), np.percentile(errorsRel, USE_PERCENTIL), np.max(errorsRel)))
        print("SYSTEM: ", labelConversion(k))
        print(tabulate(tabulateData, floatfmt=".2f", headers=["video", "#measurements", "mean [km/h]", "median [km/h]", "%.0f percentil [km/h]" %
                                                              USE_PERCENTIL, "worst [km/h]", "mean [%]", "median [%]", "%.0f percentil [%%]" % USE_PERCENTIL, "worst [%]"]))
        print()
        results[k]["TOTAL"] = {"abs": getErrorsStats(
            errorsAbs), "rel": getErrorsStats(errorsRel)}

    print()
    print()
    return outputAbs, outputRel, outputSign, results


def showDistanceMeasurementErrors(systemsData):
    print("DISTANCE ERROR STATS (ONLY TOWARDS FIRST VP)")
    outputSign = {}
    results = {}
    for k in RUN_FOR_SYSTEMS:
        results[k] = {}
        data = systemsData[k]
        errorsRel = []
        errorsAbs = []
        errorsSign = []
        tabulateData = []
        for videoId in RUN_FOR_VIDEOS:
            videoData = data[videoId]
            videoErrorsRel = []
            videoErrorsAbs = []
            videoErrorsSign = []
            scalesStats = []
            scale = videoData["data"]["camera_calibration"]["scale"]

            def projector(p): return getWorldCoordinagesOnRoadPlane(p, videoData["videoInfo"]["focal"],
                                                                    videoData["videoInfo"]["roadPlane"],
                                                                    videoData["videoInfo"]["pp"])
            for it in videoData["gtData"]["distanceMeasurement"]:
                if it["toVP1"]:
                    expectedDistance = it["distance"]
                    points = list(map(projector, [it["p1"], it["p2"]]))
                    unnormalizedDistance = np.linalg.norm(
                        points[0] - points[1])
                    distance = scale * unnormalizedDistance
                    correctScale = expectedDistance / unnormalizedDistance
                    scalesStats.append(correctScale)

                    videoErrorsAbs.append(abs(expectedDistance - distance))
                    videoErrorsRel.append(
                        abs(expectedDistance - distance) / float(expectedDistance) * 100)
                    videoErrorsSign.append(distance - expectedDistance)
            errorsRel += videoErrorsRel
            errorsAbs += videoErrorsAbs
            errorsSign += videoErrorsSign
            results[k][videoId] = {"abs": getErrorsStats(
                videoErrorsAbs), "rel": getErrorsStats(videoErrorsRel)}
            tabulateData.append((("%s %s" % videoId).replace("session", "s"), len(videoErrorsAbs),
                                 np.mean(videoErrorsAbs), np.median(videoErrorsAbs), np.percentile(
                                     videoErrorsAbs, USE_PERCENTIL), np.max(videoErrorsAbs),
                                 np.mean(videoErrorsRel), np.median(videoErrorsRel), np.percentile(videoErrorsRel, USE_PERCENTIL), np.max(videoErrorsRel), "%.4f, %.4f" % (np.mean(scalesStats), scale)))
        tabulateData.append(("TOTAL", len(errorsAbs), np.mean(errorsAbs), np.median(errorsAbs), np.percentile(errorsAbs, USE_PERCENTIL), np.max(
            errorsAbs), np.mean(errorsRel), np.median(errorsRel), np.percentile(errorsRel, USE_PERCENTIL), np.max(errorsRel), ""))
        print("SYSTEM: ", labelConversion(k))
        print(tabulate(tabulateData, floatfmt=".2f", headers=["video", "#measurements", "mean [m]", "median [m]", "%.0f percentil [m]" %
                                                              USE_PERCENTIL, "worst [m]", "mean [%]", "median [%]", "%.0f percentil [%%]" % USE_PERCENTIL, "worst [%]", "correctScale, usedScale"]))
        print()
        outputSign[k] = errorsSign
        results[k][("TOTAL")] = {"abs": getErrorsStats(
            errorsAbs), "rel": getErrorsStats(errorsRel)}

    print()
    print()
    return outputSign, results


def showFalsePositives(systemsData):
    print("FALSE POSITIVE STATS")
    tabulateData = []
    for k in RUN_FOR_SYSTEMS:
        data = systemsData[k]
        totalFalsePositives = sum(
            [i["falsePositives"] for i in iter(data.values())])
        tabulateData.append((labelConversion(k), totalFalsePositives))
    print(tabulate(tabulateData, headers=("system", "false positives")))


def showRecalls(systemsData):
    print("RECALL STATS")
    tabulateData = []
    for k in RUN_FOR_SYSTEMS:
        data = systemsData[k]
        totalFalsePositives = np.mean(
            [computeRecall(i["matches"]) for i in iter(data.values())])
        tabulateData.append((labelConversion(k), totalFalsePositives))
    print(tabulate(tabulateData, headers=("system", "mean recall")))


def showPureCamCalibErrors(systemsData):
    print("CALIBRATION ERRORS (ratios diffs)")
    results = {}
    for k in RUN_FOR_SYSTEMS:
        results[k] = {}
        data = systemsData[k]
        tabulateData = []
        allCalibAbsErrors = []
        allCalibRelErrors = []
        print("SYSTEM:", labelConversion(k))
        for sessionId, recordingId in RUN_FOR_VIDEOS:
            absCalibErrors = data[(sessionId, recordingId)]["absCalibErrors"]
            relCalibErrors = data[(sessionId, recordingId)]["relCalibErrors"]
            currentRow = [("%s %s" % (sessionId, recordingId)).replace(
                "session", "s"), len(absCalibErrors)]
            currentRow += getErrorsStats(absCalibErrors)
            currentRow += getErrorsStats(relCalibErrors)
            allCalibAbsErrors += absCalibErrors
            allCalibRelErrors += relCalibErrors
            results[k][(sessionId, recordingId)] = {"abs": getErrorsStats(
                absCalibErrors), "rel": getErrorsStats(relCalibErrors)}
            tabulateData.append(currentRow)
        tabulateData.append(["TOTAL", len(allCalibAbsErrors)] + getErrorsStats(
            allCalibAbsErrors) + getErrorsStats(allCalibRelErrors))
        results[k]["TOTAL"] = {"abs": getErrorsStats(
            allCalibAbsErrors), "rel": getErrorsStats(allCalibRelErrors)}
        print(tabulate(tabulateData, floatfmt=".2f", headers=["video", "#measurements", "mean ", "median ", "%.0f percentil " % USE_PERCENTIL, "worst ",
                                                              "mean [%]", "median [%]", "%.0f percentil [%%]" % USE_PERCENTIL, "worst [%]"]))
        print()
    print()
    print()
    return results


def showScaleCamCalibErrors(systemsData):
    print("CALIBRATION ERRORS INCLUDING SCALE")
    results = {}
    for k in RUN_FOR_SYSTEMS:
        results[k] = {}
        data = systemsData[k]
        tabulateData = []
        allScaleAbsErrors = []
        allScaleRelErrors = []
        print("SYSTEM:", labelConversion(k))
        for sessionId, recordingId in RUN_FOR_VIDEOS:
            absScaleErrors = data[(sessionId, recordingId)]["absScaleErrors"]
            relScaleErrors = data[(sessionId, recordingId)]["relScaleErrors"]
            currentRow = [("%s %s" % (sessionId, recordingId)).replace(
                "session", "s"), len(absScaleErrors)]
            currentRow += getErrorsStats(absScaleErrors)
            currentRow += getErrorsStats(relScaleErrors)
            allScaleAbsErrors += absScaleErrors
            allScaleRelErrors += relScaleErrors
            tabulateData.append(currentRow)
            results[k][(sessionId, recordingId)] = {"abs": getErrorsStats(
                absScaleErrors), "rel": getErrorsStats(relScaleErrors)}
        tabulateData.append(["TOTAL", len(allScaleAbsErrors)] + getErrorsStats(
            allScaleAbsErrors) + getErrorsStats(allScaleRelErrors))
        print(tabulate(tabulateData, floatfmt=".2f", headers=["video", "#measurements", "mean [m]", "median [m]", "%.0f percentil [m]" % USE_PERCENTIL, "worst [m]",
                                                              "mean [%]", "median [%]", "%.0f percentil [%%]" % USE_PERCENTIL, "worst [%]"]))
        results[k]["TOTAL"] = {"abs": getErrorsStats(
            allScaleAbsErrors), "rel": getErrorsStats(allScaleRelErrors)}
        print()
    print()
    print()
    return results


# %%
"""
##############################################################
########### CAMERA CALIBRATION EVALUATION ####################
##############################################################
"""


def evalCamCalibWithScale(distances, projector, scale):
    relErrors = []
    absErrors = []
    for it in distances:
        realDist = it["distance"]
        measuredDist = scale * \
            np.linalg.norm(projector(it["p1"]) - projector(it["p2"]))
        absErrors.append(abs(measuredDist - realDist))
        relErrors.append(abs(measuredDist - realDist) / realDist * 100)

    return relErrors, absErrors


def evalPureCalibration(distances, projector):
    relErrors = []
    absErrors = []

    for ind1, ind2 in itertools.combinations(list(range(len(distances))), 2):
        imageDist1 = np.linalg.norm(
            projector(distances[ind1]["p1"]) - projector(distances[ind1]["p2"]))
        imageDist2 = np.linalg.norm(
            projector(distances[ind2]["p1"]) - projector(distances[ind2]["p2"]))
        imageRatio = imageDist1 / imageDist2
        realRatio = distances[ind1]["distance"] / distances[ind2]["distance"]
        relErrors.append(abs(imageRatio - realRatio) / realRatio * 100)
        absErrors.append(abs(imageRatio - realRatio))
    return relErrors, absErrors


def getErrorsStats(errors):
    return [np.mean(errors), np.median(errors), np.percentile(errors, USE_PERCENTIL), np.max(errors)]


# %%
"""
##############################################################
######################### PLOTTING ###########################
##############################################################
"""


def initFig(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    return fig


def showSaveFig(filename, args):
    if args.saveFigures:
        filename = "figure_" + filename
        print("Saving to %s" % filename)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.03)
    if not args.noShow:
        plt.show()


def showErrorCumHistogram(data, prefix, unit, xlimMax=None):
    maxError = max(reduce(operator.add, iter(data.values())))
    bins = np.linspace(0, maxError + 1, 1000, endpoint=True)
    for k in RUN_FOR_SYSTEMS:
        errors = data[k]
        hist = np.histogram(errors, bins=bins)[0] / len(errors)
        hist = np.cumsum(hist)
        plt.plot(bins[:-1], hist, label=labelConversion(k),
                 **plotStyleCumulativeHist(k))
        plt.gca().set_xscale("log")
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.grid(which="major", axis="both",
             linewidth=0.5, linestyle="--", alpha=0.3)
    plt.grid(which="minor", axis="both",
             linewidth=0.5, linestyle="--", alpha=0.15)
    plt.ylim(0, 1)
    plt.xlim(0.1, maxError if xlimMax is None else min(maxError, xlimMax))
    plt.plot([3, 3], [0, 1], ls="--", color="grey", linewidth=2)

    plt.xlabel("Error [%s]" % (unit))
    plt.ylabel("portion of vehicles")
    plt.title("Cumulative histogram of %s errors" % (prefix))
    plt.tight_layout()
    return maxError


def showErrorHistogram(data, prefix, unit, binStep=0.5):
    xMax = int(round(max(reduce(operator.add, iter(data.values()))) + 2))
    xMin = int(round(min(reduce(operator.add, iter(data.values()))) - 2))
    xMax = max(abs(xMax), abs(xMin))
    xMin = -xMax
    bins = np.arange(xMin, xMax, step=binStep) + binStep / 2
    yMax = 0
    i = 0
    f, axs = plt.subplots(len(data), sharex=True, sharey=True, figsize=(7, 4))
    for k in RUN_FOR_SYSTEMS:
        errors = data[k]
        hist = np.histogram(errors, bins)[0] / len(errors)
        axs[i].plot(bins[1:] - binStep / 2, hist,
                    label=labelConversion(k), **plotStyleErrorHist(k))
        axs[i].plot([0, 0], [0, 1], ls="--", color="gray")
        # axs[i].legend()
        yMax = max(yMax, np.max(hist))
        i += 1

    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    positiveXticks = np.arange(0, xMax, max(1, int(xMax / 5)))
    negativeXticks = -positiveXticks[1:]
    negativeXticks = negativeXticks[::-1]

    axs[0].set_title("Histograms of %s errors" % prefix)
    plt.xticks(list(negativeXticks) + list(positiveXticks))
    plt.ylim(0, min(yMax + 0.15, 1))
    plt.xlim(xMin, xMax)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.tight_layout()
    f.subplots_adjust(hspace=0.3)
    plt.xlabel("Error [%s]" % (unit))


# %%
"""
##############################################################
########################### MAIN #############################
##############################################################
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate speed measurement systems', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c,--config',
                        dest="config", type=str, default=DEFAULT_CONFIG,
                        help="Path to config file")
    parser.add_argument('-rc,--recompute-cache',
                        dest="recomputeCache", action="store_const",
                        const=True, default=False,
                        help="Recompute cache with results")
    parser.add_argument('-m', '--mode',
                        dest="mode", type=str, default="median",
                        help="Modes of speed measurement, available modes: %s" % str(tuple(MEASUREMENT_MODES)))
    parser.add_argument("--save-figures",
                        dest="saveFigures", action="store_const",
                        const=True, default=False,
                        help="Save figures with results")
    parser.add_argument("--no-show",
                        dest="noShow", action="store_const",
                        const=True, default=False,
                        help="Do not show figures")

    args = parser.parse_args()

    MEASUREMENT_MODE = args.mode
    assert MEASUREMENT_MODE in MEASUREMENT_MODES
    RESULTS_CACHE_FILE = os.path.join(
        RESULTS_DIR, "resultsCache_%s.pkl" % MEASUREMENT_MODE)

    print(("Using custom config: %s" % args.config))
    exec(compile(open(args.config, "rb").read(), args.config, 'exec'))
    print("MEASUREMENT_MODE: ", MEASUREMENT_MODE)
    print("RUN_FOR_VIDEOS: ", RUN_FOR_VIDEOS)
    RUN_FOR_SYSTEMS = list(RUN_FOR_SYSTEMS)
    print("RUN_FOR_SYSTEMS: ", RUN_FOR_SYSTEMS)

    if not os.path.exists(RESULTS_CACHE_FILE) or args.recomputeCache:
        systemsData = {}
        for system in RUN_FOR_SYSTEMS:
            print("Computing matches with gt for system %s" % (system))
            systemsData[system] = {}
            for sessionId, recordingId in RUN_FOR_VIDEOS:
                def pTran(p): return os.path.join(
                    getPathForRecording(sessionId, recordingId), p)
                with open(os.path.join(RESULTS_DIR, "%s_%s" % (sessionId, recordingId), "system_%s.json" % system)) as f:
                    data = json.load(f)
                    gtData = loadCache(pTran("gt_data.pkl"))
                    prefilterData(data, gtData)
                    videoInfo, errorsCount = calculateSpeeds(
                        sessionId, recordingId, data, gtData, system)
                    matches = computeMatches(
                        gtData, data, sessionId, recordingId, system)
                    falsePositives = computeFalsePositives(
                        gtData, matches, data) + errorsCount

                    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(data["camera_calibration"]["vp1"],
                                                                                   data["camera_calibration"]["vp2"],
                                                                                   data["camera_calibration"]["pp"])

                    def projector(p): return getWorldCoordinagesOnRoadPlane(
                        p, focal, roadPlane, pp)
                    relScaleErrors, absScaleErrors = evalCamCalibWithScale(
                        gtData["distanceMeasurement"], projector, data["camera_calibration"]["scale"])
                    relCalibErrors, absCalibErrors = evalPureCalibration(
                        gtData["distanceMeasurement"], projector)

                    systemsData[system][(sessionId, recordingId)] = {"matches": matches,
                                                                     "falsePositives": falsePositives,
                                                                     "data": data,
                                                                     "videoInfo": videoInfo,
                                                                     "gtData": gtData,
                                                                     "relScaleErrors": relScaleErrors,
                                                                     "absScaleErrors": absScaleErrors,
                                                                     "relCalibErrors": relCalibErrors,
                                                                     "absCalibErrors": absCalibErrors}
        print("Saving results to file %s" % RESULTS_CACHE_FILE)
        saveCache(RESULTS_CACHE_FILE, systemsData)
    else:
        print("Loading results from file %s" % RESULTS_CACHE_FILE)
        systemsData = loadCache(RESULTS_CACHE_FILE)
        for system in RUN_FOR_SYSTEMS:
            if system not in systemsData:
                print((
                    " !!!! ERROR !!!!  System %s not found in cache. Please regenerate it (-rc argument)." % system))
                sys.exit(1)
            if set(systemsData[system].keys()) != set(RUN_FOR_VIDEOS):
                print((" !!!! ERROR !!!!  Mismatch in videos for system %s" % system))
                print(("Expected: %s" % str(set(RUN_FOR_VIDEOS))))
                print(("In cache: %s" % str(set(systemsData[system].keys()))))
                print("Please regenerate the cache (-rc argument)")
                sys.exit(1)
    print()

    pureCalibResults = showPureCamCalibErrors(systemsData)
    scaleResults = showScaleCamCalibErrors(systemsData)
    distErrorSign, scaleVP1Results = showDistanceMeasurementErrors(systemsData)

    absErrors, relErrors, signErrors, speedResults = showErrorStats(
        systemsData)
    showFalsePositives(systemsData)
    print()
    print()
    showRecalls(systemsData)

    print()

    fig = initFig()
    absMaxError = showErrorCumHistogram(absErrors, "absolute", "km/h")
    showSaveFig("speed_cum_error_histogram_abs.pdf", args)

    fig = initFig()
    relMaxError = showErrorCumHistogram(relErrors, "relative", "%")
    showSaveFig("speed_cum_error_histogram_rel.pdf", args)

    #showErrorHistogram(signErrors, "speed", "km/h", 1)
    #showSaveFig("error_histogram_speed.pdf", args)
    #showErrorHistogram(distErrorSign, "distance", "m", 0.5)
    #showSaveFig("error_histogram_distance.pdf", args)
