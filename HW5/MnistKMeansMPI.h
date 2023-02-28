#pragma once
#include "KMeansMPI.h"

template <int k, int pix>
class MnistKMeansMPI : public KMeansMPI<k, pix> {
public:
    void fit(vector<array<unsigned char, pix>> images, int n) {
        vector<Element> data;
        for (const auto& image : images) {
            Element element;
            std::copy(image.begin(), image.end(), element.begin());
            data.push_back(element);
        }

        KMeansMPI<k,pix>::fit(data.data(), n);
    }

protected:
    typedef std::array<u_char, pix> Element;

    /**
     *   The distance between 2 images is calculated as follows
     *   distance(x, y) = sqrt((x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_784 - y_784)^2)
    **/
    double distance(const Element& a, const Element& b) const override {
        double sum_sq_diff = 0.0;
        for (int i = 0; i < pix; i++) {
            double diff = (double)a[i] - (double)b[i];
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff);
    }
};
