package main 
 
func SortArray(array []int) []int {
    if len(array) == 0 {
        return array
    }
    if len(array) == 1 {
        return array
    }
    if len(array) == 2 {
        if array[0] > array[1] {
            return []int{array[1], array[0]}
        }
        return array
    }
    if len(array) == 3 {
        if array[0] > array[1] {
            if array[1] > array[2] {
                return []int{array[2], array[1], array[0]}
            }
            if array[0] > array[2] {
                return []int{array[2], array[0], array[1]}
            }
            return []int{array[0], array[2], array[1]}
        }
        if array[0] > array[2] {
            return []int{array[2], array[0], array[1]}
        }
        if array[1] > array[2] {
            return []int{array[2], array[1], array[0]}
        }
        return array
    }
    if len(array) == 4 {
        if array[0] > array[1] {
            if array[1] > array[2] {
                if array[2] > array[3] {
                    return []int{array[3], array[2], array[1], array[0]}
                }
                if array[1] > array[3] {
                    return []int{array[3], array[1], array[2], array[0]}
                }
                return []int{array[1], array[3], array[2], array[0]}
            }
            if array[0] > array[2] {
                if array[2] > array[3] {
                    return []int{array[3], array[2], array[0], array[1]}
                }
                return []int{array[2], array[3], array[0], array[1]}
            }
            if array[0] > array[3] {
                return []int{array[3], array[0], array[2], array[1]}
            }
            return []int{array[0], array[3], array[2], array[1]}
        }
        if array[0] > array[2] {
            if array[1] > array[2] {
                if array[2] > array[3] {
                    return []int{array[3], array[2], array[1], array[0]}
                }
                return []int{array[2], array[3], array[1], array[0]}
            }
            if array[0] > array[3] {
                return []int{array[3], array[0], array[1], array[2]}
            }
            return []int{array[0], array[3], array[1], array[2]}
        }
        if array[0] > array[3] {
            if array[1] > array[3] {
                return []int{array[3], array[1], array[0], array[2]}
            }
            return []int{array[1], array[3], array[0], array[2]}
        }
        if array[1] > array[2] {
            if array[1] > array[3] {
                return []int{array[3