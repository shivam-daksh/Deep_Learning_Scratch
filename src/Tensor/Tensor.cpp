#include "Tensor.hh"
#include <iostream>

template <typename DType>
TensorStorage<DType>::TensorStorage() {

        }

template <typename DType>
void TensorStorage<DType>::resize(size_t arraySize) {
            this->buffer.resize(arraySize);
            // for(size_t i = 0; i < arraySize; i++){
            //     this->buffer[i] = (double)0;
            // }
        }

template <typename DType>
DType& TensorStorage<DType>::operator[](const size_t index) {
            if(index >= buffer.size()) throw std::out_of_range("Tensor index out of bounds");
            return this->buffer[index];
        }

template <typename DType>
const DType& TensorStorage<DType>::operator[](const size_t index) const {
            if(index >= buffer.size()) throw std::out_of_range("Tensor index out of bounds");
            return this->buffer[index];
        }

template <typename DType>
Tensor<DType>::Tensor() {}

template <typename DType>
Tensor<DType>::Tensor(const std::vector<size_t> &shape) {
            this->resize(shape);
        }

template <typename DType>
void Tensor<DType>::resize(const std::vector<size_t> shape) {
            this->shape = shape;
            size_t arraySize = 1;
            for(size_t i = 0; i < shape.size(); i++){
                arraySize *= (shape[i]);
            }

            //Allocate (resize) the memory in the storage
            this->storage.resize(arraySize);
            this->size = arraySize;

            //update the stride
            size_t strideSize = shape.size();
            this->stride.resize(strideSize);

            if(strideSize == 0){
                return;
            }

            // Stride of the lowest dimension is 1. 
            size_t prev = 1;
            int n = static_cast<int>(strideSize);
            for(int i = n - 1; i >= 0; i--){
                if(shape[i] == 0){
                    stride[i] = 0;
                } else {
                    stride[i] = prev;
                    prev *= shape[i];
                }
            }
        }

template <typename DType>
Tensor<DType>& Tensor<DType>::fill(const DType& value) {
            for(size_t i = 0; i < this->size; i++) {
                this->storage[i] = value;
            }
            return *this;
        }

template <typename DType>
DType& Tensor<DType>::operator()(const std::vector<size_t>&indices) {
            // bound checking
            if(indices.size() != this->shape.size()) throw std::invalid_argument("Incorrect number of indices");
            for(size_t i = 0; i < indices.size(); i++) if(indices[i] >= this->shape[i]) throw std::out_of_range("Tensor index out of bounds");

            size_t index = 0;
            for(size_t i = 0; i < indices.size(); i++){
                index += (this->stride[i] * indices[i]);
            }

            return this->storage[index];
        }

template <typename DType>
const DType& Tensor<DType>::operator()(const std::vector<size_t>&indices) const {
            // bound checking
            if(indices.size() != this->shape.size()) throw std::invalid_argument("Incorrect number of indices");
            for(size_t i = 0; i < indices.size(); i++) if(indices[i] >= this->shape[i]) throw std::out_of_range("Tensor index out of bounds");

            size_t index = 0;
            for(size_t i = 0; i < indices.size(); i++){
                index += (this->stride[i] * indices[i]);
            }
            return this->storage[index];
        }

template <typename DType>
const DType& Tensor<DType>::getIndexMem(const size_t index) const{
            if(index >= this->size){
                throw std::out_of_range("Tensor index out of bounds");
            }
            return this->storage[index];
}

template <typename DType>
void Tensor<DType>::setIndexMem(const size_t index, const DType& value){
            if(index >= this->size){
                throw std::out_of_range("Tensor index out of bounds");
            }
            this->storage[index] = value;
}


template <typename DType>
bool Tensor<DType>::areShapesEqual(const std::vector<size_t>&a, const std::vector<size_t>&b) const {
            if(a.size() != b.size()) return false;
            for(int i = 0; i < a.size(); i++) if(a[i] != b[i]) return false;
            return true;
        }

template <typename DType>
bool Tensor<DType>::areBroadcastCompatible(const Tensor<DType>& a, const Tensor<DType>& b) const {
            int offset = 0;
            int n = 0;
            if(a.shape.size() > b.shape.size()){
                n = b.shape.size();
                offset = a.shape.size() - b.shape.size();

                for(int i = n-1; i >= 0; i--){
                    if(b.shape[i] != 1 && a.shape[i + offset] != 1 && b.shape[i] != a.shape[i + offset]) return false;
                }
                return true;
            } else {
                n = a.shape.size();
                offset = b.shape.size() - a.shape.size();

                for(int i = n-1; i >= 0; i--){
                    if(b.shape[i+offset] != 1 && a.shape[i] != 1 && a.shape[i] != b.shape[i+offset]) return false;
                }
                return true;
            }
        }

template <typename DType>
std::vector<size_t> Tensor<DType>::broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
            // check if any of them is scalar
            if(a.size() == 0) return b;
            if(b.size() == 0) return a;

            if(a.size() > b.size()){
                size_t offset = a.size() - b.size();
                std::vector<size_t> res;
                for(int i = b.size() - 1; i >=0; i--){
                    if((b[i] == a[offset + i]) || (b[i] == 1) || (a[offset + i] == 1)){
                        if(b[i] == a[offset + i]){
                            res.push_back(b[i]);
                        } else if(b[i] == 1) {
                            res.push_back(a[offset + i]);
                        } else {
                            res.push_back(b[i]);
                        }

                    } else {
                        throw std::invalid_argument("Tensor shapes must be compatible for broadcasting");
                    }
                }
                if(offset > 0){
                    for(int i = offset-1; i >=0; i--){
                        res.push_back(a[i]);
                    }
                }
                std::reverse(res.begin(), res.end());
                return res;
            } else {
                size_t offset = b.size() - a.size();
                std::vector<size_t> res;
                for(int i = a.size() - 1; i >=0; i--){
                    if((a[i] == b[offset + i]) || (a[i] == 1) || (b[offset + i] == 1)){
                        if(b[i + offset] == a[i]){
                            res.push_back(b[i+offset]);
                        } else if(b[i] == 1) {
                            res.push_back(a[i]);
                        } else {
                            res.push_back(b[offset + i]);
                        }
                    } else {
                        throw std::invalid_argument("Tensor shapes must be compatible for broadcasting");
                    }
                }

                if(offset > 0){
                    for(int i = offset - 1; i >=0; i--){
                        res.push_back(b[i]);
                    }
                }

                std::reverse(res.begin(), res.end());
                return res;
            }
        }

template <typename DType>
std::vector<size_t> Tensor<DType>::broadcastIndex(const std::vector<size_t>& outputIndex, const Tensor<DType>& tensor) const {
            size_t n = tensor.shape.size();
            size_t offset = 0;
            if(outputIndex.size() > n) offset = outputIndex.size() - n; //outputIndex size couldn't be smaller than the given tensor because outputIndex will always have the dimensions equal to the highest dimension tensor.
            
            std::vector<size_t> res;
            for(int i = n-1; i >= 0; i--){
                size_t value = tensor.shape[i] == 1 ? 0 : outputIndex[i+offset];
                res.push_back(value);
            }
            // for(int i = offset  - 1; i >= 0; i--){
            //     res.push_back(outputIndex[i]);
            // }

            std::reverse(res.begin(), res.end());
            return res;
        }

template <typename DType>
std::vector<size_t> Tensor<DType>::returnIndex(size_t index, const std::vector<size_t>& shape) const {
            std::vector<size_t> indices(shape.size(),0);

            for(int i = shape.size() - 1; i >= 0; i--){
                indices[i] = index % shape[i];
                index /= shape[i];
            }

            // std::reverse(indices.begin(), indices.end());

            return indices;
        }

template <typename DType>
Tensor<DType> Tensor<DType>::operator+(const Tensor<DType>& other) const {
            Tensor res;
            //check the shape
            // if(this->shape.size() != other.shape.size()) throw std::invalid_argument("Tensor size must match for addition");
            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);

                // element-wise addition
                for(size_t i = 0; i < this->size; i++){
                    res.storage[i] = this->storage[i] + other.storage[i];
                }
                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(this->shape, other.shape);
                
                res.resize(outputShape);
                for(size_t i = 0; i < res.size; i++){
                    //convert the linear index to dimension coordinates of the tensor
                    std::vector<size_t> indices = this->returnIndex(i, outputShape);
                    
                    // calculate the broadcast indices of this tensor
                    std::vector<size_t> thisIndex = this->broadcastIndex(indices, *this);
                    //calculate the broadcast indices of the other tensor
                    std::vector<size_t> otherIndex = this->broadcastIndex(indices, other);

                    // calculate the value and store
                    res(indices) = (*this)(thisIndex) + other(otherIndex);
                }
                return res;
            } else {
                std::cout << "A Shape : ";
                for(int i = 0; i < this->shape.size(); i++) std::cout << this->shape[i] << " ";
                std::cout << std::endl;
                std::cout << "B Shape : ";
                for(int i = 0; i < other.shape.size(); i++) std::cout << other.shape[i] << " ";
                std::cout << std::endl;
                throw std::invalid_argument("Tensors shape are invalid for addition");
            }
        }

template <typename DType>
Tensor<DType> Tensor<DType>::operator-(const Tensor<DType>& other) const {
            Tensor res;
            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);

                // element-wise subtraction
                for(size_t i = 0; i < this->size; i++){
                    res.storage[i] = this->storage[i] - other.storage[i];
                }
                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(this->shape, other.shape);
                
                res.resize(outputShape);
                for(size_t i = 0; i < res.size; i++){
                    //convert the linear index to dimension coordinates of the tensor
                    std::vector<size_t> indices = this->returnIndex(i, outputShape);
                    
                    // calculate the broadcast indices of this tensor
                    std::vector<size_t> thisIndex = this->broadcastIndex(indices, *this);
                    //calculate the broadcast indices of the other tensor
                    std::vector<size_t> otherIndex = this->broadcastIndex(indices, other);

                    // calculate the value and store
                    res(indices) = (*this)(thisIndex) - other(otherIndex);
                }
                return res;
            } else {
                std::cout << "A Shape : ";
                for(int i = 0; i < this->shape.size(); i++) std::cout << this->shape[i] << " ";
                std::cout << std::endl;
                std::cout << "B Shape : ";
                for(int i = 0; i < other.shape.size(); i++) std::cout << other.shape[i] << " ";
                std::cout << std::endl;
                throw std::invalid_argument("Invalid shape of the tensors element-wise subtraction");
            }
        }

template <typename DType>
Tensor<DType> Tensor<DType>::operator*(const Tensor<DType>& other) const {
            Tensor res;

            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);
                // element-wise multiplication
                for(size_t i = 0; i < this->size; i++){
                    res.storage[i] = this->storage[i] * other.storage[i];
                }
                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(this->shape, other.shape);
                
                res.resize(outputShape);
                for(size_t i = 0; i < res.size; i++){
                    //convert the linear index to dimension coordinates of the tensor
                    std::vector<size_t> indices = this->returnIndex(i, outputShape);
                    
                    // calculate the broadcast indices of this tensor
                    std::vector<size_t> thisIndex = this->broadcastIndex(indices, *this);
                    //calculate the broadcast indices of the other tensor
                    std::vector<size_t> otherIndex = this->broadcastIndex(indices, other);

                    // calculate the value and store
                    res(indices) = (*this)(thisIndex) * other(otherIndex);
                }
                return res;
            } else {
                std::cout << "A Shape : ";
                for(int i = 0; i < this->shape.size(); i++) std::cout << this->shape[i] << " ";
                std::cout << std::endl;
                std::cout << "B Shape : ";
                for(int i = 0; i < other.shape.size(); i++) std::cout << other.shape[i] << " ";
                std::cout << std::endl;
                throw std::invalid_argument("Invalid shape of the tensors element-wise multiplication");
            }
        }

template <typename DType>
Tensor<DType> Tensor<DType>::operator/(const Tensor<DType>& other) const {
            Tensor res;

            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);

                for(size_t i = 0; i < this->size; i++){
                    if(other.storage[i] == (double)0){
                        throw std::domain_error("Division by zero");
                    }

                    res.storage[i] = this->storage[i] / other.storage[i];
                }

                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(this->shape, other.shape);

                res.resize(outputShape);

                for(size_t i = 0; i < res.size; i++){
                    std::vector<size_t> indices =
                        this->returnIndex(i, outputShape);

                    std::vector<size_t> thisIndex =
                        this->broadcastIndex(indices, *this);

                    std::vector<size_t> otherIndex =
                        this->broadcastIndex(indices, other);

                    DType denominator = other(otherIndex);

                    if(denominator == (double)0){
                        throw std::domain_error("Division by zero");
                    }

                    res(indices) =
                        (*this)(thisIndex) / denominator;
                }

                return res;

            } else {
                throw std::invalid_argument(
                    "Invalid shape of the tensors division"
                );
            }
        }

template <typename DType>
Tensor<DType> Tensor<DType>::transpose() const {
            Tensor<DType> res = *this;
            std::reverse(res.shape.begin(), res.shape.end());
            std::reverse(res.stride.begin(), res.stride.end());
            return res;
        }

template <typename DType>
Tensor<DType> Tensor<DType>::T() const {
            return this->transpose();
        }

template <typename DType>
Tensor<DType> Tensor<DType>::sum(const size_t axis, const bool keepdim) const {
            if(axis >= this->shape.size()) throw std::out_of_range("Axis (index) out of the bound of the shape");
            // if(this->shape[axis] == 0) throw std::invalid_argument("Zero axis can't be reduced");
            
            //allocate new space;
            Tensor res;
            // Get the dimensions and size of the new output tensor
            std::vector<size_t> outputShape;
            for(size_t i = 0; i < this->shape.size(); i++){
                if(i == axis) {
                    if(keepdim){
                        outputShape.push_back(1);
                    }
                    continue;
                }
                outputShape.push_back(this->shape[i]);
            }
            
            if(this->shape[axis] == 0){
                throw std::invalid_argument("Zero axis can't be reduced");
            }

            // allocate the space for new tensor
            res.resize(outputShape);

            // now, reduce the tensor
            for(size_t i = 0; i < this->size; i++){
                std::vector<size_t> indices = this->returnIndex(i, this->shape);
                std::vector<size_t> outputIndex;
                for(size_t i = 0; i < indices.size(); i++){
                    if(i == axis){
                        if(keepdim){
                            outputIndex.push_back(0);
                        }
                    } else {
                        outputIndex.push_back(indices[i]);
                    }
                }

                res(outputIndex) += (*this)(indices);
            }

            return res;

        }

template <typename DType>
Tensor<DType> Tensor<DType>::mean(const size_t axis, const bool keepdim) const {
            if(axis >= this->shape.size())
                throw std::out_of_range(
                    "Axis (index) out of the bound of the shape"
                );

            if(this->shape[axis] == 0)
                throw std::invalid_argument(
                    "Zero axis can't be reduced"
                );

            Tensor res = this->sum(axis, keepdim);

            double divisor = static_cast<double>(this->shape[axis]);

            for(size_t i = 0; i < res.size; i++) {
                res.storage[i] /= divisor;
            }

            return res;
        }

template <typename DType>
Tensor<DType> Tensor<DType>::matmul(const Tensor<DType>& other) const {
    // ---------------------------------------------------------
    // 1. DIMENSION PROMOTION (Handle 1D vectors)
    // ---------------------------------------------------------
    bool this_is_1d = (this->shape.size() == 1);
    bool other_is_1d = (other.shape.size() == 1);

    if (this_is_1d || other_is_1d) {
        Tensor<DType> A = *this;
        Tensor<DType> B = other;

        // Promote A: [K] -> [1, K]
        if (this_is_1d) {
            A.shape.insert(A.shape.begin(), 1);
            A.stride.insert(A.stride.begin(), A.stride.empty() ? 1 : A.stride[0]);
        }
        // Promote B: [K] -> [K, 1]
        if (other_is_1d) {
            B.shape.push_back(1);
            B.stride.push_back(1);
        }

        // Recursively call matmul with the new 2D promoted tensors
        Tensor<DType> result = A.matmul(B);

        // Demote the result shape based on what we promoted
        if (this_is_1d && other_is_1d) {
            // [1, 1] -> [] (Scalar)
            result.shape.clear();
            result.stride.clear();
        } else if (this_is_1d) {
            // [..., 1, N] -> [..., N]
            result.shape.erase(result.shape.end() - 2);
            result.stride.erase(result.stride.end() - 2);
        } else if (other_is_1d) {
            // [..., M, 1] -> [..., M]
            result.shape.pop_back();
            result.stride.pop_back();
        }

        return result;
    }

    // Both tensors must have at least 2 dimensions
            if(this->shape.size() < 2 || other.shape.size() < 2) {
                throw std::invalid_argument(
                    "matmul requires both tensors to have at least 2 dimensions"
                );
            }

            // A: [..., M, K]
            // B: [..., K, N]

            size_t M = this->shape[this->shape.size() - 2];
            size_t K = this->shape[this->shape.size() - 1];

            size_t otherK = other.shape[other.shape.size() - 2];
            size_t N = other.shape[other.shape.size() - 1];

            // Contracting dimensions must match
            if(K != otherK) {
                throw std::invalid_argument(
                    "Incompatible dimensions for matrix multiplication"
                );
            }

            // Extract batch shapes
            std::vector<size_t> thisBatchShape(
                this->shape.begin(),
                this->shape.end() - 2
            );

            std::vector<size_t> otherBatchShape(
                other.shape.begin(),
                other.shape.end() - 2
            );

            // Find output batch shape
            std::vector<size_t> outputBatchShape =
                this->broadcastShape(thisBatchShape, otherBatchShape);

            // Output shape = broadcasted batch shape + [M,N]
            std::vector<size_t> outputShape = outputBatchShape;
            outputShape.push_back(M);
            outputShape.push_back(N);

            Tensor result(outputShape);

            // Number of matrices in the output batch
            size_t batchSize = 1;

            for(size_t dim : outputBatchShape) {
                batchSize *= dim;
            }

            // For every output matrix
            for(size_t batch = 0; batch < batchSize; batch++) {

                // Convert linear batch index → batch coordinates
                std::vector<size_t> batchIndex =
                    this->returnIndex(batch, outputBatchShape);

                // Map output batch coordinates to each input tensor
                std::vector<size_t> thisBatchIndex =
                    this->broadcastIndex(batchIndex, thisBatchShape);

                std::vector<size_t> otherBatchIndex =
                    this->broadcastIndex(batchIndex, otherBatchShape);

                // Perform ordinary matrix multiplication
                for(size_t i = 0; i < M; i++) {

                    for(size_t j = 0; j < N; j++) {

                        for(size_t k = 0; k < K; k++) {

                            // Build A index
                            std::vector<size_t> thisIndex = thisBatchIndex;

                            thisIndex.push_back(i);
                            thisIndex.push_back(k);

                            // Build B index
                            std::vector<size_t> otherIndex = otherBatchIndex;

                            otherIndex.push_back(k);
                            otherIndex.push_back(j);

                            // Build output index
                            std::vector<size_t> resultIndex =
                                batchIndex;

                            resultIndex.push_back(i);
                            resultIndex.push_back(j);

                            result(resultIndex) += (*this)(thisIndex) * other(otherIndex);
                        }
                    }
                }
            }

            return result;
        }


template class TensorStorage<double>;
template class TensorStorage<float>;
template class Tensor<double>;
template class Tensor<float>;
