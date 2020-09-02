package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

//Save Saves weights to hidden.weights & output.weights
func Save(net Network) {
	network, err := os.Create("weights/default.network")
	defer network.Close()
	if err == nil {
		network.WriteString(strconv.Itoa(net.inputs) + "\n")
		network.WriteString(strconv.Itoa(net.hiddens) + "\n")
		network.WriteString(strconv.Itoa(net.outputs) + "\n")
		network.WriteString(fmt.Sprintf("%f", net.rate) + "\n")
	}
	h, err := os.Create("weights/hidden.weights")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}

	o, err := os.Create("weights/output.weights")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

//SaveAs saves weights to str
func SaveAs(net Network, str string) {
	network, err := os.Create("weights/" + str + ".network")
	defer network.Close()
	if err == nil {
		network.WriteString(strconv.Itoa(net.inputs) + "\n")
		network.WriteString(strconv.Itoa(net.hiddens) + "\n")
		network.WriteString(strconv.Itoa(net.outputs) + "\n")
		network.WriteString(fmt.Sprintf("%f", net.rate) + "\n")
	}
	h, err := os.Create("weights/" + str + " - hidden.weights")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}

	o, err := os.Create("weights/" + str + " - output.weights")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

//Load load from hidden.weights and output.weights
func Load(net *Network) {
	network, err := ioutil.ReadFile("weights/default.network")
	if err == nil {
		data := strings.Split(string(network), "\n")
		net.inputs, _ = strconv.Atoi(data[0])
		net.hiddens, _ = strconv.Atoi(data[1])
		net.outputs, _ = strconv.Atoi(data[2])
		net.rate, _ = strconv.ParseFloat(data[3], 64)
		fmt.Printf("Loaded Default\nNeural Net: \nInputs : %d\nHiddens : %d\nOutputs : %d\nLearning Rate : %f\n", net.inputs, net.hiddens, net.outputs, net.rate)
	}
	h, err := os.Open("weights/hidden.weights")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("weights/output.weights")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
}

//LoadAs load from given str
func LoadAs(net *Network, str string) {
	network, err := ioutil.ReadFile("weights/" + str + ".network")
	if err == nil {
		data := strings.Split(string(network), "\n")
		net.inputs, _ = strconv.Atoi(data[0])
		net.hiddens, _ = strconv.Atoi(data[1])
		net.outputs, _ = strconv.Atoi(data[2])
		net.rate, _ = strconv.ParseFloat(data[3], 64)
		fmt.Printf("Loaded %s\nNeural Net: \nInputs : %d\nHiddens : %d\nOutputs : %d\nLearning Rate : %f\n", str, net.inputs, net.hiddens, net.outputs, net.rate)
	}
	h, err := os.Open("weights/" + str + " - hidden.weights")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("weights/" + str + " - output.weights")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
}
