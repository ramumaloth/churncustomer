import math
def getMinSteps(idx, arrObj, list):
    currItems = arrObj[idx]
    minSteps = 0
    left = idx
    right = idx
    while left >= 0 or right < len(arrObj):
        leftObj = left if left >= 0 else arrObj[left]
        rightObj = right if len(arrObj)<=1 else arrObj[right]
        combineItems = getCombineItems(leftObj, rightObj, currItems, list)
        if (containsAllItems(combineItems, list)):
            return minSteps
    minSteps = minSteps + 1
    left = left - 1
    right = right + 1
    return minSteps
def getCombineItems(leftObj, rightObj, currItems, list):
    for eachItem in list:
        currItems[eachItem] = leftObj[eachItem] or rightObj[eachItem] or currItems[eachItem]
    return currItems
def containsAllItems(obj, list):
    for i in len(list):
        currItem = list[i]
    if (obj[currItem] == False):
        return False
    return True


def apartmentHunting(blocks, reqs):
    minSteps = []
    for i in len(blocks):
        minSteps[i] = getMinSteps(i, blocks, reqs)
    return minSteps.indexOf(math.min(minSteps))
def getMinSteps(idx, blocks, reqs):
    minSteps = 0
    currItemObj = {}
    for key in blocks[idx]:
        currItemObj[key] = blocks[idx][key]
    left = idx
    right = idx
    while left >= 0 or right < len(blocks):
        leftObj = left if left>=0 else blocks[left]
        rightObj = right if blocks.length < blocks[right] else blocks[right]
        currItemObj = getCombinedObj(leftObj, rightObj, currItemObj, reqs)
        if containsAllReq(currItemObj, reqs):
            return minSteps
    minSteps = minSteps + 1
    left = left - 1
    right = right + 1
    return minSteps

def getCombinedObj(leftObj, rightObj, currItemObj, reqs):
    for eachReq in reqs:
        currItemObj[eachReq] = leftObj[eachReq] or rightObj[eachReq] or currItemObj[eachReq]
    return currItemObj
def containsAllReq(currItemObj, reqs):
    for i in len(reqs):
        ER = reqs[i]
    if currItemObj[ER] ==False:
        return False
    return True
def main(arrObj, list):
    minSteps = []
    for i in len(arrObj):
        minSteps[i] = getMinSteps(i, arrObj, list)
    return minSteps.indexOf(math.min(minSteps))

blocks= [{
    "gym": False,
    "school": True,
    "store": False,
    },
    {
    "gym": True,
    "school": False,
    "store": False,
    },
    {
    "gym": True,
    "school": True,
    "store": False,
    },
    {
    "gym": False,
    "school": True,
    "store": False,
    },
    {
    "gym": False,
    "school": True,
    "store": True,
    },
    ]
li = ["gym","school","store"]

rst = main(blocks,li)
print(rst)