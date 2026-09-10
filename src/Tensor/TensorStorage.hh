// Tensor Storage
#include <algorithm>
#include <vector>
#include <stdexcept>
class TensorStorage {
    private:
        std::vector<double> buffer;
    public:
        TensorStorage(){

        }
        void resize(size_t arraySize){
            this->buffer.resize(arraySize);
            for(size_t i = 0; i < arraySize; i++){
                this->buffer[i] = 0.0;
            }
        }

        double& operator[](const size_t index){
            if(index >= buffer.size()) throw std::out_of_range("Tensor index out of bounds");
            return this->buffer[index];
        }

        const double& operator[](const size_t index) const{
            if(index >= buffer.size()) throw std::out_of_range("Tensor index out of bounds");
            return this->buffer[index];
        }
};

class Tensor {
    private:
        TensorStorage storage;
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t size;
    public:
        Tensor(){}
        Tensor(const std::vector<size_t> &shape){
            this->resize(shape);
        }

        void resize(const std::vector<size_t> shape){
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
        
        double& operator()(const std::vector<size_t>&indices){
            // bound checking
            if(indices.size() != this->shape.size()) throw std::invalid_argument("Incorrect number of indices");
            for(size_t i = 0; i < indices.size(); i++) if(indices[i] >= this->shape[i]) throw std::out_of_range("Tensor index out of bounds");

            size_t index = 0;
            for(size_t i = 0; i < indices.size(); i++){
                index += (this->stride[i] * indices[i]);
            }

            return this->storage[index];
        }
        
        const double& operator()(const std::vector<size_t>&indices) const{
            // bound checking
            if(indices.size() != this->shape.size()) throw std::invalid_argument("Incorrect number of indices");
            for(size_t i = 0; i < indices.size(); i++) if(indices[i] >= this->shape[i]) throw std::out_of_range("Tensor index out of bounds");

            size_t index = 0;
            for(size_t i = 0; i < indices.size(); i++){
                index += (this->stride[i] * indices[i]);
            }
            return this->storage[index];
        }

        bool areShapesEqual(const std::vector<size_t>&a, const std::vector<size_t>&b)const {
            if(a.size() != b.size()) return false;
            for(int i = 0; i < a.size(); i++) if(a[i] != b[i]) return false;
            return true;
        }

        bool areBroadcastCompatible(const Tensor& a, const Tensor& b) const {
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

        std::vector<size_t> broadcastShape(const Tensor& a, const Tensor& b) const{
            // check if any of them is scalar
            if(a.shape.size() == 0) return b.shape;
            if(b.shape.size() == 0) return a.shape;

            if(a.shape.size() > b.shape.size()){
                size_t offset = a.shape.size() - b.shape.size();
                std::vector<size_t> res;
                for(int i = b.shape.size() - 1; i >=0; i--){
                    if((b.shape[i] == a.shape[offset + i]) || (b.shape[i] == 1) || (a.shape[offset + i] == 1)){
                        if(b.shape[i] == a.shape[offset + i]){
                            res.push_back(b.shape[i]);
                        } else if(b.shape[i] == 1) {
                            res.push_back(a.shape[offset + i]);
                        } else {
                            res.push_back(b.shape[i]);
                        }

                    } else {
                        throw std::invalid_argument("Tensor shapes must be compatible for broadcasting");
                    }
                }
                if(offset > 0){
                    for(int i = offset-1; i >=0; i--){
                        res.push_back(a.shape[i]);
                    }
                }
                std::reverse(res.begin(), res.end());
                return res;
            } else {
                size_t offset = b.shape.size() - a.shape.size();
                std::vector<size_t> res;
                for(int i = a.shape.size() - 1; i >=0; i--){
                    if((a.shape[i] == b.shape[offset + i]) || (a.shape[i] == 1) || (b.shape[offset + i] == 1)){
                        if(b.shape[i + offset] == a.shape[i]){
                            res.push_back(b.shape[i+offset]);
                        } else if(b.shape[i] == 1) {
                            res.push_back(a.shape[i]);
                        } else {
                            res.push_back(b.shape[offset + i]);
                        }
                    } else {
                        throw std::invalid_argument("Tensor shapes must be compatible for broadcasting");
                    }
                }

                if(offset > 0){
                    for(int i = offset - 1; i >=0; i--){
                        res.push_back(b.shape[i]);
                    }
                }

                std::reverse(res.begin(), res.end());
                return res;
            }
        }

        std::vector<size_t> broadcastIndex(const std::vector<size_t>& outputIndex, const Tensor& tensor) const {
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

        std::vector<size_t> returnIndex(size_t index, const std::vector<size_t>& shape) const {
            std::vector<size_t> indices(shape.size(),0);

            for(int i = shape.size() - 1; i >= 0; i--){
                indices[i] = index % shape[i];
                index /= shape[i];
            }

            // std::reverse(indices.begin(), indices.end());

            return indices;
        }
        // Element-wise Addition
        Tensor operator+(const Tensor& other) const{
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
                std::vector<size_t> outputShape = this->broadcastShape(*this, other);
                
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
        
        // Element-wise Subtraction
        Tensor operator-(const Tensor& other) const{
            Tensor res;
            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);

                // element-wise subtraction
                for(size_t i = 0; i < this->size; i++){
                    res.storage[i] = this->storage[i] - other.storage[i];
                }
                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(*this, other);
                
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

        Tensor operator*(const Tensor& other) const{
            Tensor res;

            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);
                // element-wise multiplication
                for(size_t i = 0; i < this->size; i++){
                    res.storage[i] = this->storage[i] * other.storage[i];
                }
                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(*this, other);
                
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



        Tensor operator/(const Tensor& other) const{
            Tensor res;

            if(this->areShapesEqual(this->shape, other.shape)){
                res.resize(this->shape);

                for(size_t i = 0; i < this->size; i++){
                    if(other.storage[i] == 0.0){
                        throw std::domain_error("Division by zero");
                    }

                    res.storage[i] = this->storage[i] / other.storage[i];
                }

                return res;

            } else if(areBroadcastCompatible(*this, other)){
                std::vector<size_t> outputShape = this->broadcastShape(*this, other);

                res.resize(outputShape);

                for(size_t i = 0; i < res.size; i++){
                    std::vector<size_t> indices =
                        this->returnIndex(i, outputShape);

                    std::vector<size_t> thisIndex =
                        this->broadcastIndex(indices, *this);

                    std::vector<size_t> otherIndex =
                        this->broadcastIndex(indices, other);

                    double denominator = other(otherIndex);

                    if(denominator == 0.0){
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

        Tensor transpose() {
            std::reverse(this->shape.begin(), this->shape.end());
            std::reverse(this->stride.begin(), this->stride.end());
            return *this;
        }

        Tensor T() {
            return this->transpose();
        }

        Tensor sum(size_t axis =  0, bool keepdim=false){
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

        
};