// CSCI 114 Final Project
// Hannah Lee
// Final Project

// Short description about my project
// LA Metro Bike Share
// can we predict if someone is a casual
// rider or a subscriber based on how they use the bikes?
// Using KNN to classify Walk-up vs Monthly Pass riders

// only using standard C++ libraries - no external dependencies needed
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
// needed to get day of week
#include <ctime>
// for iota
#include <numeric>
// for shuffle
#include <random>

using namespace std;

// stores just the fields I need from each trip
struct BikeTrip {
    // trip length in seconds
    int duration;
    // 0-23, hour the trip started
    int hour;
    // 0=Monday through 6=Sunday
    int day;
    // 1 if round trip, 0 if one way
    int roundtrip;
    // 1 = Monthly Pass, 0 = Walk-up
    int label;
};

// splits a csv line by commas but ignores commas inside quotes
vector<string> splitLine(const string& line) {
    vector<string> fields;
    string cur;
    bool inQuotes = false;
    for (char c : line) {
        // toggle quote mode
        if (c == '"') inQuotes = !inQuotes;
        // only split on commas outside quotes
        else if (c == ',' && !inQuotes) {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    // push the last field
    fields.push_back(cur);
    return fields;
}

// extracts hour and day of week from datetime string
// format looks like "2017-01-19T17:05:00.000"
bool parseTime(const string& s, int& hour, int& dow) {
    // string too short, skip it
    if (s.size() < 16) return false;
    try {
        // characters 11-12 are the hour
        hour = stoi(s.substr(11, 2));
        // fill a tm struct so mktime can figure out the weekday
        struct tm t = {};
        // year since 1900
        t.tm_year  = stoi(s.substr(0, 4)) - 1900;
        // month is 0-indexed
        t.tm_mon   = stoi(s.substr(5, 2)) - 1;
        t.tm_mday  = stoi(s.substr(8, 2));
        // let system figure out DST
        t.tm_isdst = -1;
        // this fills in tm_wday for us
        mktime(&t);
        // tm_wday is 0=Sunday, convert to 0=Monday
        dow = (t.tm_wday == 0) ? 6 : t.tm_wday - 1;
    } catch (...) {
        // if anything fails just skip the row
        return false;
    }
    return true;
}

// reads the csv and builds a vector of BikeTrip structs
// only keeps Walk-up and Monthly Pass rows
vector<BikeTrip> loadData(const string& filename) {
    vector<BikeTrip> trips;
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "couldn't open file: " << filename << endl;
        // return empty vector
        return trips;
    }

    string line;
    // first line is the header, skip it
    getline(file, line);

    while (getline(file, line)) {
        // skip blank lines
        if (line.empty()) continue;
        auto fields = splitLine(line);
        // need at least 14 columns
        if (fields.size() < 14) continue;

        // column 13 is Passholder Type
        string passholder = fields[13];
        if (passholder != "Walk-up" && passholder != "Monthly Pass") continue;

        int hour = 0, dow = 0;
        // column 2 is Start Time
        if (!parseTime(fields[2], hour, dow)) continue;

        BikeTrip t;
        try {
            // column 1 is Duration in seconds
            t.duration = stoi(fields[1]);
        } catch (...) {
            // skip if duration is missing
            continue;
        }

        t.hour      = hour;
        t.day       = dow;
        // column 12 is Trip Route Category
        t.roundtrip = (fields[12] == "Round Trip") ? 1 : 0;
        // this is our target class label
        t.label     = (passholder == "Monthly Pass") ? 1 : 0;
        trips.push_back(t);
    }

    cout << "loaded " << trips.size() << " trips" << endl;
    return trips;
}

// scales each feature to 0-1 so duration doesnt dominate KNN distance
vector<double> normalize(const BikeTrip& t) {
    return {
        // duration range is 60s to 86400s
        (t.duration - 60.0) / (86400.0 - 60.0),
        // hour range is 0-23
        t.hour / 23.0,
        // day range is 0-6
        t.day  / 6.0,
        // already 0 or 1
        (double)t.roundtrip
    };
}

// euclidean distance between two feature vectors
double dist(const vector<double>& a, const vector<double>& b) {
    double sum = 0;
    for (int i = 0; i < (int)a.size(); i++)
        // sum of squared differences
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

// finds k nearest neighbors and returns majority vote label
int knn(const vector<double>& query,
        const vector<vector<double>>& trainF,
        const vector<int>& trainL,
        int k) {
    // compute distance from query to every training point
    vector<pair<double,int>> dists;
    for (int i = 0; i < (int)trainF.size(); i++)
        dists.push_back({ dist(query, trainF[i]), trainL[i] });
    // partial sort is faster since we only need the top k
    partial_sort(dists.begin(), dists.begin() + k, dists.end());
    // count votes from the k nearest neighbors
    int votes = 0;
    for (int i = 0; i < k; i++) votes += dists[i].second;
    // more than half voted 1, return 1
    return (votes > k / 2) ? 1 : 0;
}

// writes an SVG chart file that opens in any browser
// y2 and label2 are optional - pass empty string for label2 to draw one line only
void writeSVG(const string& filename,
              const string& title,
              const string& xlabel,
              const string& ylabel,
              const vector<double>& x,
              const vector<double>& y1,
              const vector<double>& y2,
              const string& label1,
              const string& label2,
              const vector<string>& xlabels = {}) {

    // chart dimensions
    int W = 800, H = 500;
    int marginL = 70, marginR = 30, marginT = 60, marginB = 80;
    int chartW = W - marginL - marginR;
    int chartH = H - marginT - marginB;

    // find max y value for scaling
    double maxY = 0;
    for (double v : y1) maxY = max(maxY, v);
    if (!label2.empty())
        for (double v : y2) maxY = max(maxY, v);
    maxY *= 1.1;

    ofstream f(filename);
    f << "<svg width='" << W << "' height='" << H << "' xmlns='http://www.w3.org/2000/svg'>\n";
    // white background
    f << "<rect width='" << W << "' height='" << H << "' fill='white'/>\n";

    // title
    f << "<text x='" << W/2 << "' y='35' text-anchor='middle' "
      << "font-family='Arial' font-size='16' font-weight='bold'>" << title << "</text>\n";

    // x axis label
    f << "<text x='" << (marginL + chartW/2) << "' y='" << (H-10) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13'>" << xlabel << "</text>\n";

    // y axis label (rotated)
    f << "<text x='15' y='" << (marginT + chartH/2) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13' "
      << "transform='rotate(-90,15," << (marginT + chartH/2) << ")'>" << ylabel << "</text>\n";

    // vertical axis line
    f << "<line x1='" << marginL << "' y1='" << marginT
      << "' x2='" << marginL << "' y2='" << (marginT+chartH)
      << "' stroke='black' stroke-width='1'/>\n";

    // horizontal axis line
    f << "<line x1='" << marginL << "' y1='" << (marginT+chartH)
      << "' x2='" << (marginL+chartW) << "' y2='" << (marginT+chartH)
      << "' stroke='black' stroke-width='1'/>\n";

    int n = x.size();
    double xStep = (double)chartW / n;

    // draw y1 as line (blue = Walk-up or single series)
    f << "<polyline fill='none' stroke='steelblue' stroke-width='2' points='";
    for (int i = 0; i < n; i++) {
        double px = marginL + i * xStep + xStep/2;
        double py = marginT + chartH - (y1[i] / maxY) * chartH;
        f << px << "," << py << " ";
    }
    f << "'/>\n";

    // draw y2 as line (orange) - only if label2 is not empty
    if (!label2.empty()) {
        f << "<polyline fill='none' stroke='orange' stroke-width='2' points='";
        for (int i = 0; i < n; i++) {
            double px = marginL + i * xStep + xStep/2;
            double py = marginT + chartH - (y2[i] / maxY) * chartH;
            f << px << "," << py << " ";
        }
        f << "'/>\n";
    }

    // draw dots on y1 line
    for (int i = 0; i < n; i++) {
        double px = marginL + i * xStep + xStep/2;
        double py = marginT + chartH - (y1[i] / maxY) * chartH;
        f << "<circle cx='" << px << "' cy='" << py << "' r='3' fill='steelblue'/>\n";
    }

    // draw dots on y2 line - only if label2 is not empty
    if (!label2.empty()) {
        for (int i = 0; i < n; i++) {
            double px = marginL + i * xStep + xStep/2;
            double py = marginT + chartH - (y2[i] / maxY) * chartH;
            f << "<circle cx='" << px << "' cy='" << py << "' r='3' fill='orange'/>\n";
        }
    }

    // x tick labels
    for (int i = 0; i < n; i++) {
        double px = marginL + i * xStep + xStep/2;
        string lbl = xlabels.empty() ? to_string((int)x[i]) : xlabels[i];
        // only label every other tick if there are many
        if (n > 10 && i % 2 != 0) continue;
        f << "<text x='" << px << "' y='" << (marginT+chartH+18) << "' "
          << "text-anchor='middle' font-family='Arial' font-size='11'>" << lbl << "</text>\n";
    }

    // y tick labels - 5 evenly spaced ticks
    for (int i = 0; i <= 5; i++) {
        double val = maxY * i / 5;
        double py  = marginT + chartH - (val / maxY) * chartH;
        // tick label
        f << "<text x='" << (marginL-5) << "' y='" << (py+4) << "' "
          << "text-anchor='end' font-family='Arial' font-size='11'>"
          << (int)(val*10)/10.0 << "</text>\n";
        // light grid line
        f << "<line x1='" << marginL << "' y1='" << py
          << "' x2='" << (marginL+chartW) << "' y2='" << py
          << "' stroke='#ddd' stroke-width='0.5'/>\n";
    }

    // legend - always show label1
    f << "<rect x='" << (W-marginR-160) << "' y='" << (marginT+10)
      << "' width='12' height='12' fill='steelblue'/>\n";
    f << "<text x='" << (W-marginR-144) << "' y='" << (marginT+21)
      << "' font-family='Arial' font-size='12'>" << label1 << "</text>\n";

    // only show label2 if it exists
    if (!label2.empty()) {
        f << "<rect x='" << (W-marginR-160) << "' y='" << (marginT+30)
          << "' width='12' height='12' fill='orange'/>\n";
        f << "<text x='" << (W-marginR-144) << "' y='" << (marginT+41)
          << "' font-family='Arial' font-size='12'>" << label2 << "</text>\n";
    }

    f << "</svg>\n";
    f.close();
    cout << "saved " << filename << endl;
}

// plot 1 - duration distribution
// walk-up riders tend to take much longer trips
void plotDuration(const vector<BikeTrip>& trips) {
    // cap at 1 hour so the histogram is readable
    const int CAP  = 3600;
    const int BINS = 30;
    double binW = CAP / (double)BINS;

    vector<double> wuBins(BINS, 0), mpBins(BINS, 0);
    int wuTotal = 0, mpTotal = 0;

    for (auto& t : trips) {
        // cap the duration
        int d = min(t.duration, CAP);
        // figure out which bin it falls in
        int b = min((int)(d / binW), BINS-1);
        if (t.label == 0) { wuBins[b]++; wuTotal++; }
        else              { mpBins[b]++; mpTotal++; }
    }

    // convert counts to percentages so the two groups are comparable
    vector<double> x(BINS), wuPct(BINS), mpPct(BINS);
    for (int i = 0; i < BINS; i++) {
        // x axis in minutes
        x[i]     = i * binW / 60.0;
        wuPct[i] = 100.0 * wuBins[i] / wuTotal;
        mpPct[i] = 100.0 * mpBins[i] / mpTotal;
    }

    writeSVG("plot1_duration.svg",
             "Trip Duration Distribution by Rider Type",
             "Duration (minutes, capped at 60)",
             "% of Trips",
             x, wuPct, mpPct,
             "Walk-up", "Monthly Pass");
}

// plot 2 - trips by hour of day
// monthly pass riders peak at 8am and 5pm (commuters)
void plotHour(const vector<BikeTrip>& trips) {
    // count trips per hour for each rider type
    vector<double> wuHr(24, 0), mpHr(24, 0);
    int wuTotal = 0, mpTotal = 0;

    for (auto& t : trips) {
        if (t.label == 0) { wuHr[t.hour]++; wuTotal++; }
        else              { mpHr[t.hour]++; mpTotal++; }
    }

    // convert to percentages
    vector<double> x(24), wuPct(24), mpPct(24);
    for (int i = 0; i < 24; i++) {
        x[i]     = i;
        wuPct[i] = 100.0 * wuHr[i] / wuTotal;
        mpPct[i] = 100.0 * mpHr[i] / mpTotal;
    }

    writeSVG("plot2_hour.svg",
             "Trips by Hour of Day",
             "Hour of Day",
             "% of Trips",
             x, wuPct, mpPct,
             "Walk-up", "Monthly Pass");
}

// plot 3 - trips by day of week
// expecting subscribers to ride more on weekdays
void plotDay(const vector<BikeTrip>& trips) {
    vector<double> wuDay(7, 0), mpDay(7, 0);
    int wuTotal = 0, mpTotal = 0;

    for (auto& t : trips) {
        if (t.label == 0) { wuDay[t.day]++; wuTotal++; }
        else              { mpDay[t.day]++; mpTotal++; }
    }

    // convert to percentages
    vector<double> x(7), wuPct(7), mpPct(7);
    for (int i = 0; i < 7; i++) {
        x[i]     = i;
        wuPct[i] = 100.0 * wuDay[i] / wuTotal;
        mpPct[i] = 100.0 * mpDay[i] / mpTotal;
    }

    vector<string> days = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
    writeSVG("plot3_day.svg",
             "Trips by Day of Week",
             "Day of Week",
             "% of Trips",
             x, wuPct, mpPct,
             "Walk-up", "Monthly Pass", days);
}

// plot 4 - one way vs round trip by rider type
void plotRoute(const vector<BikeTrip>& trips) {
    // OW = one way, RT = round trip
    int wuOW=0, wuRT=0, mpOW=0, mpRT=0;

    for (auto& t : trips) {
        if (t.label == 0) { if (t.roundtrip) wuRT++; else wuOW++; }
        else              { if (t.roundtrip) mpRT++; else mpOW++; }
    }

    // totals for percentage calc
    double wuTot = wuOW + wuRT;
    double mpTot = mpOW + mpRT;

    // percentages for each category
    double wuOWpct = 100.0 * wuOW / wuTot;
    double wuRTpct = 100.0 * wuRT / wuTot;
    double mpOWpct = 100.0 * mpOW / mpTot;
    double mpRTpct = 100.0 * mpRT / mpTot;

    // write a bar chart SVG manually
    // two groups (One Way, Round Trip), two bars each (Walk-up, Monthly Pass)
    int W = 800, H = 500;
    int marginL = 70, marginR = 30, marginT = 60, marginB = 80;
    int chartW = W - marginL - marginR;
    int chartH = H - marginT - marginB;

    // max y is just over 100 since percentages add to 100
    double maxY = 110.0;

    // bar dimensions
    double groupW = chartW / 2.0;  // width per group
    double barW   = groupW * 0.28; // width of each individual bar
    double gap    = groupW * 0.08; // gap between bars in a group

    // x positions for each bar
    double owWU = marginL + groupW * 0.5 - barW - gap / 2;
    double owMP = owWU + barW + gap;
    double rtWU = marginL + groupW * 1.5 - barW - gap / 2;
    double rtMP = rtWU + barW + gap;

    ofstream f("plot4_route.svg");
    f << "<svg width='" << W << "' height='" << H << "' xmlns='http://www.w3.org/2000/svg'>\n";
    // white background
    f << "<rect width='" << W << "' height='" << H << "' fill='white'/>\n";

    // title
    f << "<text x='" << W/2 << "' y='35' text-anchor='middle' "
      << "font-family='Arial' font-size='16' font-weight='bold'>"
      << "Route Type by Rider Type</text>\n";

    // x axis label
    f << "<text x='" << (marginL + chartW/2) << "' y='" << (H-10) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13'>Route Type</text>\n";

    // y axis label
    f << "<text x='15' y='" << (marginT + chartH/2) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13' "
      << "transform='rotate(-90,15," << (marginT + chartH/2) << ")'>% of Trips</text>\n";

    // vertical axis line
    f << "<line x1='" << marginL << "' y1='" << marginT
      << "' x2='" << marginL << "' y2='" << (marginT + chartH)
      << "' stroke='black' stroke-width='1'/>\n";

    // horizontal axis line
    f << "<line x1='" << marginL << "' y1='" << (marginT + chartH)
      << "' x2='" << (marginL + chartW) << "' y2='" << (marginT + chartH)
      << "' stroke='black' stroke-width='1'/>\n";

    // grid lines and y tick labels
    for (int i = 0; i <= 5; i++) {
        double val = maxY * i / 5.0;
        double py  = marginT + chartH - (val / maxY) * chartH;
        f << "<line x1='" << marginL << "' y1='" << py
          << "' x2='" << (marginL + chartW) << "' y2='" << py
          << "' stroke='#ddd' stroke-width='0.5'/>\n";
        f << "<text x='" << (marginL - 5) << "' y='" << (py + 4) << "' "
          << "text-anchor='end' font-family='Arial' font-size='11'>"
          << (int)val << "</text>\n";
    }

    // helper lambda to draw one bar
    auto drawBar = [&](double xPos, double pct, const string& color) {
        double barH = (pct / maxY) * chartH;
        double yTop = marginT + chartH - barH;
        f << "<rect x='" << xPos << "' y='" << yTop
          << "' width='" << barW << "' height='" << barH
          << "' fill='" << color << "' opacity='0.85'/>\n";
        // value label on top of bar
        f << "<text x='" << (xPos + barW/2) << "' y='" << (yTop - 4) << "' "
          << "text-anchor='middle' font-family='Arial' font-size='11'>"
          << (int)(pct + 0.5) << "%</text>\n";
    };

    // draw the four bars
    drawBar(owWU, wuOWpct, "steelblue");
    drawBar(owMP, mpOWpct, "orange");
    drawBar(rtWU, wuRTpct, "steelblue");
    drawBar(rtMP, mpRTpct, "orange");

    // group labels on x axis
    double owCenter = marginL + groupW * 0.5;
    double rtCenter = marginL + groupW * 1.5;
    f << "<text x='" << owCenter << "' y='" << (marginT + chartH + 20) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13'>One Way</text>\n";
    f << "<text x='" << rtCenter << "' y='" << (marginT + chartH + 20) << "' "
      << "text-anchor='middle' font-family='Arial' font-size='13'>Round Trip</text>\n";

    // legend
    f << "<rect x='" << (W - marginR - 160) << "' y='" << (marginT + 10)
      << "' width='12' height='12' fill='steelblue' opacity='0.85'/>\n";
    f << "<text x='" << (W - marginR - 144) << "' y='" << (marginT + 21)
      << "' font-family='Arial' font-size='12'>Walk-up</text>\n";
    f << "<rect x='" << (W - marginR - 160) << "' y='" << (marginT + 30)
      << "' width='12' height='12' fill='orange' opacity='0.85'/>\n";
    f << "<text x='" << (W - marginR - 144) << "' y='" << (marginT + 41)
      << "' font-family='Arial' font-size='12'>Monthly Pass</text>\n";

    f << "</svg>\n";
    f.close();
    cout << "saved plot4_route.svg" << endl;
}

// plot 5 - knn accuracy for different values of k
// train on 80%, test on 20%
void plotKNN(const vector<BikeTrip>& trips) {
    // build normalized feature vectors for all trips
    vector<vector<double>> features;
    vector<int> labels;
    for (auto& t : trips) {
        features.push_back(normalize(t));
        labels.push_back(t.label);
    }

    // shuffle using seed 42 so results are reproducible
    mt19937 rng(42);
    vector<int> idx(trips.size());
    // fill with 0,1,2,3...
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);

    // 80% train, 20% test
    int trainN = (int)(trips.size() * 0.8);
    vector<vector<double>> trainF, testF;
    vector<int> trainL, testL;

    for (int i = 0; i < (int)idx.size(); i++) {
        if (i < trainN) {
            trainF.push_back(features[idx[i]]);
            trainL.push_back(labels[idx[i]]);
        } else {
            testF.push_back(features[idx[i]]);
            testL.push_back(labels[idx[i]]);
        }
    }

    // only use 1500 test points so it finishes in reasonable time
    int sampleSize = min(1500, (int)testF.size());

    vector<double> kVals = {1, 3, 5, 7, 9, 11, 15, 21};
    vector<double> accs;

    cout << "running KNN..." << endl;
    for (double k : kVals) {
        int correct = 0;
        for (int i = 0; i < sampleSize; i++) {
            // classify this test point
            int pred = knn(testF[i], trainF, trainL, (int)k);
            // check if we got it right
            if (pred == testL[i]) correct++;
        }
        double acc = 100.0 * correct / sampleSize;
        accs.push_back(acc);
        cout << "  k=" << (int)k << "  acc=" << acc << "%" << endl;
    }

    // pass empty vector and empty string for y2/label2 so only one line draws
    vector<double> empty;
    writeSVG("plot5_knn.svg",
             "KNN Accuracy vs K (Walk-up vs Monthly Pass)",
             "K (number of neighbors)",
             "Accuracy (%)",
             kVals, accs, empty,
             "KNN Accuracy", "");
}

int main(int argc, char* argv[]) {
    // default csv name, can also pass as command line arg
    string csvFile = "metro-bike-share-trip-data.csv";
    if (argc > 1) csvFile = argv[1];

    cout << "CSCI 114 Final Project - LA Metro Bike Share" << endl;

    vector<BikeTrip> trips = loadData(csvFile);
    // stop if nothing loaded
    if (trips.empty()) return 1;

    // print how many of each type we have
    int nWU = 0, nMP = 0;
    for (auto& t : trips) { if (t.label == 0) nWU++; else nMP++; }
    cout << "walk-up: " << nWU << "  monthly pass: " << nMP << endl << endl;

    // generate all 5 plots
    plotDuration(trips);
    plotHour(trips);
    plotDay(trips);
    plotRoute(trips);
    plotKNN(trips);

    cout << "\ndone! open the .svg files in your browser to see the plots" << endl;
    return 0;
}
