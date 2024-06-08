package domain;

import lombok.Data;

import java.util.*;

@Data
public class Node {

    public int id; // node order starting from 1
//    public List<Integer> nodeId; // used to find connected edges:node appears more than once
    public String name; // node name
    public List<String> controlFlowMarks;  // splited by +
    public List<String> dataFlowMarks;   // splited by +
    public List<String> existDataTokens;  // splited by , in {}  original Set<String>
    public String guard = "";
    public List<Integer> inputEdges; // input edge ids
    public List<Integer> outputEdges; // output edge ids

    public Node()
    {
//        nodeId = null;
        controlFlowMarks = new ArrayList<>();
        dataFlowMarks = new ArrayList<>();
        existDataTokens = new ArrayList<>();
        inputEdges = new ArrayList<>();
        outputEdges = new ArrayList<>();
    }

}
