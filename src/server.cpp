#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// JSON formatting library
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <chilitags/chilitags.hpp>

#include <cxxopts.hpp>
#include <RedisImageHelper.hpp>

using namespace std;
using namespace cv;
using namespace chilitags;

bool VERBOSE = false;
bool STREAM_MODE = false;
bool SET_MODE = false;

std::string redisInputKey = "camera0";
std::string redisOutputKey = "camera0:markers";
std::string redisInputCameraParametersKey = "camera0";

std::string redisHost = "127.0.0.1";
int redisPort = 6379;

struct contextData {
    uint width;
    uint height;
    uint channels;
    RedisImageHelperSync* clientSync;
};

static int parseCommandLine(cxxopts::Options options, int argc, char** argv)
{
    auto result = options.parse(argc, argv);
    if (result.count("h")) {
        std::cout << options.help({"", "Group"});
        return EXIT_FAILURE;
    }

    if (result.count("v")) {
        VERBOSE = true;
        std::cerr << "Verbose mode enabled." << std::endl;
    }

    if (result.count("i")) {
        redisInputKey = result["i"].as<std::string>();
        if (VERBOSE) {
            std::cerr << "Input key was set to `" << redisInputKey << "`." << std::endl;
        }
    }
    else {
        if (VERBOSE) {
            std::cerr << "No input key was specified. Input key was set to default (" << redisInputKey << ")." << std::endl;
        }
    }

    if (result.count("o")) {
        redisOutputKey = result["o"].as<std::string>();
        if (VERBOSE) {
            std::cerr << "Output key was set to `" << redisOutputKey << "`." << std::endl;
        }
    }
    else {
        if (VERBOSE) {
            std::cerr << "No output key was specified. Output key was set to default (" << redisOutputKey << ")." << std::endl;
        }
    }

    if (result.count("u")) {
        STREAM_MODE = false;
        SET_MODE = false;
        if (VERBOSE) {
            std::cerr << "Unique mode enabled." << std::endl;
        }
    }

    if (result.count("s")) {
        STREAM_MODE = true;
        if (VERBOSE) {
            std::cerr << "PUBLISH stream mode enabled." << std::endl;
        }
    }

    if (result.count("g")) {
        SET_MODE = true;
        if (VERBOSE) {
            std::cerr << "GET/SET stream mode enabled." << std::endl;
        }
    }

    if (result.count("redis-port")) {
        redisPort = result["redis-port"].as<int>();
        if (VERBOSE) {
            std::cerr << "Redis port set to " << redisPort << "." << std::endl;
        }
    }
    else {
        if (VERBOSE) {
            std::cerr << "No redis port specified. Redis port was set to " << redisPort << "." << std::endl;
        }
    }

    if (result.count("redis-host")) {
        redisHost = result["redis-host"].as<std::string>();
        if (VERBOSE) {
            std::cerr << "Redis host set to " << redisHost << "." << std::endl;
        }
    }
    else {
        if (VERBOSE) {
            std::cerr << "No redis host was specified. Redis host was set to " << redisHost << "." << std::endl;
        }
    }

    if (result.count("camera-parameters")) {
        redisInputCameraParametersKey = result["camera-parameters"].as<std::string>();
        if (VERBOSE) {
            std::cerr << "Camera parameters input key was set to " << redisInputCameraParametersKey << std::endl;
        }
    }
    else {
        if (VERBOSE) {
            std::cerr << "No camera parameters intput key specified. Camera parameters input key was set to " << redisInputCameraParametersKey << std::endl;
        }
    }

    if (!result.count("u") && !result.count("s") && !result.count("g")) {
        std::cerr << "You need to specify at least the stream method option with -u, -s or -g" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

rapidjson::Value* CTagToJSON(const std::pair<int, chilitags::Quad>& tag, rapidjson::Document::AllocatorType& allocator)
{
    rapidjson::Value* tagObj = new rapidjson::Value(rapidjson::kObjectType);
    int id = tag.first;
    int dir = 0;
    tagObj->AddMember("id", id, allocator);
    tagObj->AddMember("dir", dir, allocator);
    tagObj->AddMember("confidence", 100, allocator);
    tagObj->AddMember("type", "CTag", allocator);

    const cv::Mat_<cv::Point2f> corners(tag.second);

    cv::Point2f center = 0.5f * (corners(0) + corners(2));

    rapidjson::Value centerArray(rapidjson::kArrayType);
    centerArray.PushBack(center.x, allocator);
    centerArray.PushBack(center.y, allocator);
    tagObj->AddMember("center", centerArray, allocator);

    rapidjson::Value cornerArray(rapidjson::kArrayType);
    for (int points = 0 ; points < 4 ; ++points)
    {
        cornerArray.PushBack(corners(points).x, allocator);
        cornerArray.PushBack(corners(points).y, allocator);
    }

    tagObj->AddMember("corners", cornerArray, allocator);
    return tagObj;
}

rapidjson::Value* CTagsToJSON(chilitags::TagCornerMap* tags, rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value* tagsObj = new rapidjson::Value(rapidjson::kArrayType);
    int markersCount = tags->size();
    if (VERBOSE) {
        std::cerr << "Found " << markersCount << " Chilitags markers." << std::endl;
    }

    for (const std::pair<int, chilitags::Quad>& tag : *tags)
    {
        rapidjson::Value* tagObj = CTagToJSON(tag, allocator);
        tagsObj->PushBack(*tagObj, allocator);
        delete tagObj;
    }
    return tagsObj;
}

string process(Image* image) {
    Mat imageCv, gray;
    imageCv = Mat(image->height(), image->width(), CV_8UC3, image->data());
    cvtColor(imageCv, gray, CV_RGB2GRAY);
    chilitags::Chilitags chilitags;
    chilitags::TagCornerMap* tags = new chilitags::TagCornerMap();
    *tags = chilitags.find(gray);

    rapidjson::Document jsonMarkers;
    jsonMarkers.SetObject();
    rapidjson::Document::AllocatorType &allocator = jsonMarkers.GetAllocator();

    rapidjson::Value* markersObj = CTagsToJSON(tags, allocator);
    jsonMarkers.AddMember("markers", *markersObj, allocator);
    delete markersObj;

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    jsonMarkers.Accept(writer);
    return strbuf.GetString();
}

void onImagePublished(redisAsyncContext* c, void* rep, void* privdata) {
    redisReply *reply = (redisReply*) rep;
    if  (reply == NULL) { return; }
    if (reply->type != REDIS_REPLY_ARRAY || reply->elements != 3) {
        if (VERBOSE) {
            std::cerr << "Error: Bad reply format." << std::endl;
        }
        return;
    }

    struct contextData* data = static_cast<struct contextData*>(privdata);
    if (data == NULL) {
        if(VERBOSE) {
            std::cerr << "Error: Could not retrieve context data from private data." << std::endl;
        }
        return;
    }
    uint width = data->width;
    uint height = data->height;
    uint channels = data->channels;
    RedisImageHelperSync* clientSync = data->clientSync;

    Image* image = RedisImageHelper::dataToImage(reply->element[2]->str, width, height, channels, reply->element[2]->len);
    if (image == NULL) {
        if (VERBOSE) {
            std::cerr << "Error: Could not retrieve image from data." << std::endl;
        }
        return;
    }
    std::string json = process(image);
    if (SET_MODE) {
        clientSync->setString((char*)json.c_str(), redisOutputKey);
    }
    clientSync->publishString((char*)json.c_str(), redisOutputKey);

    if (VERBOSE) {
        std::cerr << json << std::endl;
    }
    delete image;
}

int main(int argc, char** argv) {
    cxxopts::Options options("aruco-detection-server", "Aruco markers detection server.");
    options.add_options()
            ("redis-port", "The port to which the redis client should try to connect.", cxxopts::value<int>())
            ("redis-host", "The host adress to which the redis client should try to connect", cxxopts::value<std::string>())
            ("i, input", "The redis input key where data are going to arrive.", cxxopts::value<std::string>())
            ("o, output", "The redis output key where to set output data.", cxxopts::value<std::string>())
            ("s, stream", "Activate stream mode. In stream mode the program will constantly process input data and publish output data.")
            ("u, unique", "Activate unique mode. In unique mode the program will only read and output data one time.")
            ("g, stream-set", "Enable stream get/set mode. If stream mode is already enabled setting this will cause to publish and set the same data at the same time")
            ("c, camera-parameters", "The redis input key where camera-parameters are going to arrive.", cxxopts::value<std::string>())
            ("v, verbose", "Enable verbose mode. This will print helpfull process informations on the standard error stream.")
            ("h, help", "Print this help message.");

    if (parseCommandLine(options, argc, argv)) {
        return EXIT_FAILURE;
    }

    RedisImageHelperSync clientSync(redisHost, redisPort, redisInputKey);
    if (!clientSync.connect()) {
        std::cerr << "Cannot connect to redis server. Please ensure that a redis server is up and running." << std::endl;
        return EXIT_FAILURE;
    }

    struct contextData data;
    data.width = clientSync.getInt(redisInputCameraParametersKey + ":width");
    data.height = clientSync.getInt(redisInputCameraParametersKey + ":height");
    data.channels = clientSync.getInt(redisInputCameraParametersKey + ":channels");
    if (data.width == -1 || data.height == -1 || data.channels == -1) {
        // TODO: Fix double free or corruption error when camera parameters can not be loaded.
        std::cerr << "Could not find camera parameters (width height channels). Please specify where to find them in redis with the --camera-parameters option parameters." << std::endl;
        return EXIT_FAILURE;
    }
    data.clientSync = &clientSync;

    if (STREAM_MODE) {
        RedisImageHelperAsync clientAsync(redisHost, redisPort, redisInputKey);
        if (!clientAsync.connect()) {
            std::cerr << "Cannot connect to redis server. Please ensure that a redis server is up and running." << std::endl;
            return EXIT_FAILURE;
        }
        clientAsync.subscribe(redisInputKey, onImagePublished, static_cast<void*>(&data));
    }
    else {
        bool loop = true;
        while (loop) {
            Image* image = clientSync.getImage(data.width, data.height, data.channels);
            std::string json = process(image);
            clientSync.setString((char*)json.c_str(), redisOutputKey);
            std::cerr << json << std::endl;
            delete image;
            loop = SET_MODE ? true : false;
        }
    }
}
