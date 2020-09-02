package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"time"
)

func clear() {
	cmd := exec.Command("cmd", "/c", "cls")
	cmd.Stdout = os.Stdout
	cmd.Run()
}
func mnistTraining(net *Network, rounds int) {
	rand.Seed(time.Now().UTC().UnixNano())
	then := time.Now()
	count := 0
	correct := 0
	clear()
	fmt.Println("TRAINING NEURAL NETWORK")
	for i := 0; i < rounds; i++ {
		trainingData, err := os.Open("mnist_train.csv")
		if err != nil {
			fmt.Println("error loading training data")
			break
		} else {
			r := csv.NewReader(bufio.NewReader(trainingData))
			for {
				numRow, err := r.Read()
				if err == io.EOF {
					break
				}
				inputs := make([]float64, net.inputs)
				for j := range inputs {
					x, _ := strconv.ParseFloat(numRow[j], 64)
					inputs[j] = (x / 255.0 * 0.999) + 0.001

				}
				targets := make([]float64, 10)
				for j := range targets {
					targets[j] = 0.001
				}
				x, _ := strconv.Atoi(numRow[0])
				targets[x] = 0.999
				outputs := net.Train(inputs, targets)
				best := 0
				highest := 0.0
				for i := 0; i < net.outputs; i++ {
					if outputs.At(i, 0) > highest {
						best = i
						highest = outputs.At(i, 0)
					}
				}
				answer, _ := strconv.Atoi(numRow[0])
				if best == answer {
					correct++
				}
				count++
			}
			trainingData.Close()
			fmt.Printf("round # %d \nrecord # %d Correct = %d\n", i+1, count, correct)
			fmt.Printf("Accuracy: %.2f%s\n", float64(correct)/float64(count)*100, "%")

		}
	}
	now := time.Since(then)
	fmt.Printf("Time taken: %s\n", now)
}

func mnistPredicting(net *Network) {
	then := time.Now()
	testFile, err := os.Open("mnist_test.csv")
	if err != nil {
		fmt.Println("Prediction read failed.")
		return
	}
	defer testFile.Close()
	correct := 0
	numberCount := 0
	num := csv.NewReader(bufio.NewReader(testFile))
	for {
		numRow, err := num.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(numRow[i], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputs, _ := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		answer, _ := strconv.Atoi(numRow[0])
		if best == answer {
			correct++
		}
		numberCount++

	}
	now := time.Since(then)
	fmt.Printf("Time taken: %s\n", now)
	fmt.Printf("correct: %d / %d \n", correct, numberCount)
	fmt.Printf("Accuracy: %.2f\n", float64(correct)/float64(numberCount)*100)

}

func main() {
	// 784 inputs, 1 for each pixel

	mnist := flag.String("mnist", "", "Train or predict on mnist set")
	fileName := flag.String("f", "", "Save/load specified network")
	hiddenNumbers := flag.Int("h", 200, "Specify number of hidden nodes")
	learningRate := flag.Float64("r", 0.1, "Specify number for learning rate")
	rounds := flag.Int("i", 1, "Specify rounds to train")
	flag.Parse()

	net := CreateNet(784, *hiddenNumbers, 10, *learningRate)
	switch *mnist {
	case "train":
		fmt.Println("Training Neural Network")
		mnistTraining(&net, *rounds)
		if *fileName == "" {
			fmt.Println("Saving to default")
			Save(net)
		} else {
			fmt.Println("Saving to " + *fileName)

			SaveAs(net, *fileName)
		}
	case "predict":
		if *fileName == "" {
			fmt.Println("Loading default")
			Load(&net)
			mnistPredicting(&net)
		} else {
			fmt.Println("Loading " + *fileName)
			LoadAs(&net, *fileName)
			mnistPredicting(&net)
		}
	default:
	}
}
