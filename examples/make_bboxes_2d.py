# Not for use Within the Kubric Pipeline
# Designed to be run by a separate environment
# For Now..

import json
import os
import cv2 as cv
import argparse

mefile = "metadata.json"
filestarter = "rgba_"
endpng = ".png"
folderspace = "bboxdraw"


def dostuff(**kwargs):
    workspace = None
    mepng = "0000"
    if kwargs["path"]:
        workspace = kwargs["path"]
        mefile = str(workspace) + "/metadata.json"
    if workspace is None:
        workspace = "."

    f = open(mefile)
    data = json.load(f)
    height, width = data["metadata"]["resolution"]
    num_frames = data["metadata"]["num_frames"]
    frame_count = 0

    while frame_count < num_frames:
        print(frame_count)
        if frame_count >= 10:
            mepng = "000"

        elif frame_count >= 100:
            mepng = "00"
        if kwargs["path"]:
            if filestarter:
                name = str(workspace) + "/" + filestarter + mepng + str(frame_count) + endpng
                name_no_workspace = filestarter + mepng + str(frame_count) + endpng
            else:
                name = str(workspace) + "/" + mepng + str(frame_count) + endpng
                name_no_workspace = mepng + str(frame_count) + endpng
        else:
            if filestarter:
                name = filestarter + mepng + str(frame_count) + endpng
                print(name)
            else:
                name = mepng + str(frame_count) + endpng
                print(name)

        # Open the image
        image = cv.imread(name)
        # Make a Copy
        alltogether = image.copy()

        for i in data["instances"]:
            imageblank = image.copy()
            print(f"Hey " + str(len(i["bbox_frames"])))
            try:
                ouritem = i["bbox_frames"]
                if ouritem[frame_count]:
                    print(f"YES for {frame_count}")
                if i["bbox_frames"][frame_count] is frame_count:
                    item = i["bboxes"][frame_count]
                y_min = int(item[0] * height)
                x_min = int(item[1] * width)
                y_max = int(item[2] * height)
                x_max = int(item[3] * width)

            except Exception as e:
                print(f"[EXCEPTION] {e}")
            finally:

                # you need top-left corner and bottom-right corner of rectangle
                cv.rectangle(alltogether, (x_min, y_min), (x_max, y_max), (255, 0, 0))
                cv.rectangle(imageblank, (x_min, y_min), (x_max, y_max), (255, 0, 0))
            #Label image
            if kwargs["label"]:
                cv.putText(alltogether, i["asset_id"], (x_min, y_min), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
                cv.putText(imageblank, i["asset_id"], (x_min, y_min), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

            #Save image
            if kwargs["file"]:
                if not os.path.exists(f"{workspace}\\{folderspace}"):
                    os.makedirs(f"{workspace}\\{folderspace}")
                if kwargs["path"]:
                    name = name_no_workspace
                if not os.path.exists(f"{workspace}\\{folderspace}\\{i['asset_id']}"):
                    os.makedirs(f"{workspace}\\{folderspace}\\{i['asset_id']}")
                path = f"{workspace}\\{folderspace}\\{i['asset_id']}\\"
                #print(f"{workspace}/{folderspace}/{i['asset_id']}"+name)
                cv.imwrite(path + i['asset_id'] +name, imageblank)


        if kwargs["file"]:
            if kwargs["path"]:
                name = name_no_workspace
            if not os.path.exists(f"{workspace}/{folderspace}"):
                os.makedirs(f"{workspace}/{folderspace}")
            path = f"{workspace}/{folderspace}/"
            cv.imwrite(os.path.join(path, f'{name}.jpg'), alltogether)
        frame_count += 1

        #Show image
        if kwargs["show"]:
            cv.imshow(name, alltogether)
            cv.waitKey(0)
            cv.destroyAllWindows()


    f.close()
    if frame_count == num_frames:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Label the image", action="store_true")
    parser.add_argument("--show", help="Show the image", action="store_true")
    parser.add_argument("--file", help="Save the image", action="store_true")
    parser.add_argument("--path", help="Path to the folder")
    args = parser.parse_args()
    dostuff(**vars(args))