package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestIntersection(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("NO", Intersection([2]int{1, 2}, [2]int{2, 3}))
    assert.Equal("NO", Intersection([2]int{-1, 1}, [2]int{0, 4}))
    assert.Equal("YES", Intersection([2]int{-3, -1}, [2]int{-5, 5}))
    assert.Equal("YES", Intersection([2]int{-2, 2}, [2]int{-4, 0}))
    assert.Equal("NO", Intersection([2]int{-11, 2}, [2]int{-1, -1}))
    assert.Equal("NO", Intersection([2]int{1, 2}, [2]int{3, 5}))
    assert.Equal("NO", Intersection([2]int{1, 2}, [2]int{1, 2}))
    assert.Equal("NO", Intersection([2]int{-2, -2}, [2]int{-3, -2}))
}
