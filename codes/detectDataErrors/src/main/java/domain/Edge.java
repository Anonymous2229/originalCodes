package domain;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Data
public class Edge {

    public int id; // edge order
    public int sourceId;
    public int targetId;
    public String transition;
    public List<String> readData;
    public List<String> writeData;
    public String guard;

    public Edge(){
        readData = new ArrayList<>();
        writeData = new ArrayList<>();
    }
}
