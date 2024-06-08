package domain;

import lombok.Data;

import java.util.HashSet;
import java.util.Set;

@Data
public class Place {
    public int no;  // 序号，从1开始
    public String id; // place id
    public String name; // place text
    public String type; // place color set
    public String initMark; // place initial marking
    public Set<String> incomingArcs; // input arc ids
    public Set<String> outgoingArcs; // output arc ids

    public Place(){
        incomingArcs = new HashSet<>();
        outgoingArcs = new HashSet<>();
    }

}
