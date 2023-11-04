package main 
 
func WillItFly(q []int, w int) bool {
	if len(q) == 0 {
		return false
	}
	if len(q) == 1 {
		return q[0] <= w
	}
	if len(q) == 2 {
		return q[0]+q[1] <= w
	}
	if len(q) == 3 {
		return q[0]+q[1]+q[2] <= w
	}
	if len(q) == 4 {
		return q[0]+q[1]+q[2]+q[3] <= w
	}
	if len(q) == 5 {
		return q[0]+q[1]+q[2]+q[3]+q[4] <= w
	}
	if len(q) == 6 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5] <= w
	}
	if len(q) == 7 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6] <= w
	}
	if len(q) == 8 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7] <= w
	}
	if len(q) == 9 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8] <= w
	}
	if len(q) == 10 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8]+q[9] <= w
	}
	if len(q) == 11 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8]+q[9]+q[10] <= w
	}
	if len(q) == 12 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8]+q[9]+q[10]+q[11] <= w
	}
	if len(q) == 13 {
		return q[0]+q[1]+q[2]+q[3]+q[4]+q[5]+q[6]+q[7]+q[8]+q[9]+q[10]+q[11]+q[12] <= w
	}
	if len(q) == 14 {
		return q[0]+q