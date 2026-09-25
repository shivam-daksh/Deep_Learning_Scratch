#pragma once
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>

template <typename DType>
class TensorStorage {
    private:
        std::vector<DType> buffer;
    public:
        TensorStorage();
        void resize(size_t arraySize);
        DType& operator[](const size_t index);
        const DType& operator[](const size_t index) const;
};

template <typename DType>
class Tensor {
    private:
        TensorStorage<DType> storage;
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t size;
        bool areShapesEqual(const std::vector<size_t>&a, const std::vector<size_t>&b)const;
        bool areBroadcastCompatible(const Tensor& a, const Tensor& b) const;
        std::vector<size_t> broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) const;
        std::vector<size_t> broadcastIndex(const std::vector<size_t>& outputIndex, const Tensor& tensor) const;
        std::vector<size_t> returnIndex(size_t index, const std::vector<size_t>& shape) const;
    public:
        // accessibility methods
        Tensor();
        Tensor(const std::vector<size_t> &shape);
        void resize(const std::vector<size_t> shape);
        Tensor<DType>& fill(const DType& value);
        DType& operator()(const std::vector<size_t>&indices);
        const DType& operator()(const std::vector<size_t>&indices) const;
        inline const std::vector<size_t>& getShape() const{
            return this->shape;
        }
        const size_t getSize() const{
            return this->size;
        }

        inline const std::vector<size_t>& getStride()const{
            return this->stride;
        }
        // access the actual serial storage tensor
        const DType& getIndexMem(const size_t index);
        void setIndexMem(const size_t index, const DType& value);

        // tensor arithmetic/math
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor transpose();
        Tensor T();
        Tensor sum(const size_t axis =  0, const bool keepdim=false) const;
        Tensor mean(const size_t axis = 0, const bool keepdim = false) const;
        Tensor matmul(const Tensor& other) const;
        
};
