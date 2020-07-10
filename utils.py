import cv2


def dummy(dummy):
    """
    Just a dummy function to receive the calls by cv2.getTracbarPos() internally.. (We mean internally as.. we are passing the function name(i.,e function address), to be more clear like calling another function by help of another function in C (Linux..refer GFG for clear example..))
    Does nothing..
    :param dummy: A dummy parameter, just to receive the parameter sent by cv2.getTracbarPos() internally..
    :return: nothing
    """
    pass


def createTrackbars():
    # A window to adjust the track bars..
    cv2.namedWindow("Adjust Colors by track bars..", cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("Blue", "Adjust Colors by track bars..", 0, 255, dummy)
    cv2.createTrackbar("Green", "Adjust Colors by track bars..", 0, 255, dummy)
    cv2.createTrackbar("Red", "Adjust Colors by track bars..", 0, 255, dummy)


def adjust_color_by_tackbars():
    """

    :return:
    """
    blue = cv2.getTrackbarPos("Blue", "Adjust Colors by track bars..")
    green = cv2.getTrackbarPos("Green", "Adjust Colors by track bars..")
    red = cv2.getTrackbarPos("Red", "Adjust Colors by track bars..")
    return blue, green, red
