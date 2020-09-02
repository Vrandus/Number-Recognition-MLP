package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

//Network structure for NN
type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	rate          float64
}

//CreateNet creates network n to be trained
func CreateNet(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:  input,
		hiddens: hidden,
		outputs: output,
		rate:    rate,
	}
	randHidden := randArray(net.inputs*net.hiddens, float64(net.inputs))
	randOutputs := randArray(net.outputs*net.hiddens, float64(net.hiddens))
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randHidden)
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randOutputs)
	return
}

func randArray(size int, v float64) (weights []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	weights = make([]float64, size)
	for i := 0; i < size; i++ {
		weights[i] = dist.Rand()
	}
	return
}

//Predict forward propagation
func (net Network) Predict(inputData []float64) (mat.Matrix, mat.Matrix) {
	inputMat := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := Dot(net.hiddenWeights, inputMat)
	hiddenOutputs := Apply(sigmoid, hiddenInputs)
	finalInputs := Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := Apply(sigmoid, finalInputs)
	return finalOutputs, hiddenOutputs
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

//Train forward propagate -> find errors -> back propagate
func (net *Network) Train(inputData []float64, targetData []float64) mat.Matrix {
	finalOutputs, hiddenOutputs := net.Predict(inputData)
	inputMat := mat.NewDense(len(inputData), 1, inputData)
	targetMat := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := Sub(targetMat, finalOutputs)
	hiddenErrors := Dot(net.outputWeights.T(), outputErrors)

	net.outputWeights = Add(net.outputWeights,
		Scale(net.rate,
			Dot(Multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = Add(net.hiddenWeights,
		Scale(net.rate, Dot(Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
			inputMat.T()))).(*mat.Dense)
	return finalOutputs
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	oneMat := mat.NewDense(rows, 1, o)
	return Multiply(m, Sub(oneMat, m))
}
