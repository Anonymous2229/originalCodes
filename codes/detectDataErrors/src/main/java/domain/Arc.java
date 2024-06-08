package domain;

import lombok.Data;

@Data
public class Arc {
    public int no; // 序号，从1开始
    public String id; // arc id
    public int orientation; // TtoP 0 PtoT 1 BOTHDIR 2
    public String transId; // transend idref
    public String placeId; // placeend idref
    public String annotation; // annot text
}
